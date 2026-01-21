from label_mapper import label_mapper_image_name
import os
import pickle
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import trange
import json
import numpy as np

def _ensure_colorcet_cmap():
    """
    CXAS depends on `colorcet` colormaps (cc.cm.glasbey_bw_minc_20), but in some
    environments `cc.cm.glasbey_bw_minc_20` can be `None` (e.g. missing optional
    matplotlib integration). Patch in a small fallback so CXAS can import.
    """
    try:
        import colorcet as cc
    except Exception:
        return

    cm = getattr(cc, "cm", None)
    cmap = getattr(cm, "glasbey_bw_minc_20", None) if cm is not None else None
    if callable(cmap):
        return

    import colorsys

    def fallback(i: int):
        # Deterministic HSV palette; returns RGBA floats in [0,1].
        h = (float(i) * 0.61803398875) % 1.0  # golden-ratio spacing
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        return (r, g, b, 1.0)

    if cm is None:
        class _CM:
            pass

        cc.cm = _CM()
        cm = cc.cm
    setattr(cm, "glasbey_bw_minc_20", fallback)


_ensure_colorcet_cmap()

from cxas import CXAS
import cxas.visualize as cxas_vis
import shutil


join = os.path.join

class_check_tables = [
        "thoracic spine", 
        "all vertebrae",  
        "cervical spine", 
        "lumbar spine",   
        "clavicle set",   
        "scapula set",    
        "ribs",           
        "ribs super",     
        "diaphragm",      
        "mediastinum",    
        "abdomen",        
        "heart region",   
        "breast tissue",  
        "trachea",        
        "lung zones",     
        "lung halves",    
        "vessels",        
        "lung lobes"     
    ]

need_mask_table = {
    "bone":["ribs", "ribs super"],
    "lung":["lung zones", "lung halves", "lung lobes"],
    "heart":["heart region"],
    "mediastinum":["mediastinum"],
}

class config:    
    tfms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
    #### Check table
    class_check_tables = [
        "thoracic spine", 
        "all vertebrae",  
        "cervical spine",
        "lumbar spine",   
        "clavicle set", 
        "scapula set",  
        "ribs",         
        "ribs super",   
        "diaphragm",    
        "mediastinum",  
        "abdomen",      
        "heart region", 
        "breast tissue",  
        "trachea",        
        "lung zones",     
        "lung halves",    
        "vessels",        
        "lung lobes"  
    ]

def save_as_npz_unified(path, content_dict):
    """
    Save all organ masks in ONE compressed NPZ file.
    Args:
        path: output file path
        content_dict: {organ_name: tensor, ...}
    """
    data_to_save = {name: tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor 
                    for name, tensor in content_dict.items()}
    np.savez_compressed(path, **data_to_save)

def load_from_npz_unified(path, organs=None):
    """
    Load organ masks from unified NPZ file with optional filtering.
    Args:
        path: path to unified_masks.npz file
        organs: list of organ names to load (e.g., ['bone', 'lung'])
               If None, loads all available organs
    Returns:
        dict: {organ_name: tensor, ...}
    """
    data = np.load(path, allow_pickle=False)
    all_organs = {name: torch.from_numpy(data[name]).float() for name in data.files}
    
    if organs is None:
        return all_organs
    
    # Load only specified organs
    filtered = {org: all_organs[org] for org in organs if org in all_organs}
    return filtered

def _load_paxray_label_to_index() -> dict:
    """
    Map anatomy label string -> channel index in CXAS output.

    CXAS stores masks as a [num_labels, H, W] array where channel i corresponds to
    `id2label_dict[str(i)]`. We load the same label list from `paxray_labels.json`
    shipped with this repo.
    """
    labels_path = Path(__file__).parent / "paxray_labels.json"
    with open(labels_path) as f:
        id2label_dict = json.load(f)["label_dict"]
    return {label: int(idx) for idx, label in id2label_dict.items()}

def create_cxas_model():
    return CXAS(model_name = 'UNet_ResNet50_default',
                gpus       = '0')

def get_caption(caption_path):
    with open(caption_path) as f:
        data = json.load(f)
    return data

def cxas_infer(cxas_model, path_image, out_path, args):
    """Run CXAS inference and store raw masks as .npy for downstream processing.

    For both IU-Xray and MIMIC we need the numpy output so the post-processing
    step can select organ channels and compress into unified NPZ files.
    """
    _ = cxas_model.process_file(
        filename=path_image,
        do_store=True,
        output_directory=out_path,
        storage_type="npy",
    )

    # Optional: save visualization JPGs if requested (debug/QA only)
    if getattr(args, "save_individual_jpg", False):
        _ = cxas_model.process_file(
            filename=path_image,
            do_store=True,
            output_directory=out_path,
            storage_type="jpg",
        )

    return

def segmentation_cxas_iu_xray(cxas_model, caption_data, args):
    for k in caption_data.keys():
        print("{} part has been started !!!!!!".format(k))
        for i in trange(len(caption_data[k])):
            if not os.path.exists(config.iu_xray_saved_path): os.mkdir(config.iu_xray_saved_path)
            saved_path = join(config.iu_xray_saved_path, caption_data[k][i]["id"])
            if not os.path.exists(saved_path): os.mkdir(saved_path)
            for i,each in enumerate(caption_data[k][i]["image_path"]):
                # Skip inference if unified NPZ for this view already exists (resume-friendly)
                mask_out_dir = join(saved_path, f"{i}_mask")
                unified_npz_path = join(mask_out_dir, "unified_masks.npz")
                if os.path.exists(unified_npz_path):
                    continue
                img = join(config.iu_xray_image, each)
                cxas_infer(cxas_model, img, saved_path, args)

def saved_mask_pickle_iu_xray():
    print("Processing: Saving ALL organ masks in ONE unified file per view (compressed NPZ)...")
    label_to_idx = _load_paxray_label_to_index()
    img_file_list = [join(config.iu_xray_saved_path, each.stem) for each in list(Path(config.iu_xray_saved_path).glob("*"))]
    
    for folder_path in trange(len(img_file_list)):
        patient_folder = img_file_list[folder_path]
        
        # IU-Xray typically has 2 images: 0 (frontal) and 1 (lateral)
        for view_idx in range(2):
            mask_out_dir = join(patient_folder, "{}_mask".format(view_idx))
            # Resume check: if unified file already exists, skip this view
            unified_npz_path = join(mask_out_dir, "unified_masks.npz")
            if os.path.exists(unified_npz_path):
                continue

            npy_path = join(patient_folder, "{}.npy".format(view_idx))
            if not os.path.exists(npy_path):
                continue

            full_masks = np.load(npy_path)
            if not os.path.exists(mask_out_dir): 
                os.mkdir(mask_out_dir)

            # Store all organs in ONE dictionary
            all_organs_dict = {}

            # Process each anatomical group (bone, lung, heart, etc.)
            for need_class, sub_classes in need_mask_table.items():
                if len(sub_classes) == 0: 
                    continue
                    
                selected_indices = []
                for each_class in sub_classes:
                    for label_name in label_mapper_image_name[each_class]:
                        idx = label_to_idx.get(label_name)
                        if idx is not None:
                            selected_indices.append(idx)

                if not selected_indices:
                    continue

                # Select channels and resize to 224x224
                selected = full_masks[selected_indices]
                selected_t = torch.from_numpy(selected).unsqueeze(1).float()  # [C, 1, H, W]
                resized = F.interpolate(selected_t, size=(224, 224), mode="nearest").squeeze(1)  # [C, 224, 224]
                resized = (resized > 0.5).float()

                # Store in dictionary instead of saving separately
                all_organs_dict[need_class] = resized

            # Save ALL organs in ONE file!
            # Save ALL organs in ONE file!
            save_as_npz_unified(unified_npz_path, all_organs_dict)

            # Cleanup: optionally remove raw intermediate outputs to save space.
            if not getattr(args, "keep_npy", False) and os.path.exists(npy_path):
                os.remove(npy_path)
            raw_jpg_dir = join(patient_folder, str(view_idx))
            if os.path.exists(raw_jpg_dir):
                shutil.rmtree(raw_jpg_dir)

def segmentation_cxas_mimic_cxr(cxas_model, mimic_cxr_caption, args):
    for key in mimic_cxr_caption.keys():
        print("{} part has been started !!!!!!".format(key))
        for i in trange(len(mimic_cxr_caption[key])):
            item = mimic_cxr_caption[key][i]
            item_id = item["id"]
            assert len(item["image_path"]) == 1, "{} has more than 1 images".format(item_id)
            item_image_path = item["image_path"][0]
            img_path = join(config.mimic_cxr_image, item_image_path)
            saved_path = join(config.mimic_cxr_path, "images_segmented", item_image_path.replace(".jpg",""))
            if os.path.exists(saved_path): continue
            if not os.path.exists(saved_path): os.makedirs(saved_path)
            #################### Segmentation ####################
            cxas_infer(cxas_model, img_path, saved_path, args) 

def segmentation_cxas_mimic_cxr_folder(cxas_model, input_folder, args):
    """
    Process single MIMIC-CXR folder with STREAMING NPZ conversion.
    Input:  data/mimic_cxr/images/files/p10/p10xxx/sxxxxx/*.jpg
    Output: data/mimic_cxr_segmentation/files/p10/p10xxx/sxxxxx/unified_masks.npz
    
    Streaming approach: .npy → NPZ → delete temp files immediately
    This prevents disk space exhaustion with 38k+ images per folder.
    """
    from tqdm import tqdm
    
    input_folder = Path(input_folder).resolve()
    images_root = Path(config.mimic_cxr_image).resolve()
    output_root = Path(config.mimic_cxr_saved_path).resolve()

    # Ensure the input folder is under the expected images root
    try:
        input_folder.relative_to(images_root)
    except ValueError:
        # Fallback: auto-detect images root from the provided input path (up to 'images')
        parts = input_folder.parts
        if "images" in parts:
            idx = parts.index("images")
            images_root = Path(*parts[:idx+1]).resolve()
        else:
            raise ValueError(f"Input folder must be under {images_root}, got {input_folder}")

    # Discover images recursively (supports JPG, JPEG, PNG)
    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        img_files.extend(input_folder.rglob(ext))
    
    if not img_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Processing {len(img_files)} images from {input_folder}...")
    print("Streaming mode: converting .npy to NPZ on-the-fly (saves disk space)")
    
    label_to_idx = _load_paxray_label_to_index()
    
    for img_file in tqdm(img_files):
        img_path = str(img_file)
        # Mirror the exact images path under the segmentation root
        relative_path = img_file.relative_to(images_root)
        output_folder = output_root / relative_path.parent
        output_folder.mkdir(parents=True, exist_ok=True)
        
        npz_path = output_folder / "unified_masks.npz"
        
        if npz_path.exists():
            continue
        
        temp_dir = output_folder / ".temp_cxas"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Run CXAS segmentation
            try:
                cxas_infer(cxas_model, img_path, str(temp_dir), args)
            except OSError as e:
                # Handle disk space errors gracefully
                if "requested and 0 written" in str(e) or "No space left" in str(e):
                    print(f"\n⚠️  DISK SPACE ERROR on {img_file.name} - skipping")
                    print(f"   Tip: Clear disk space or process smaller batches")
                    continue
                else:
                    raise
            
            # Find and convert .npy immediately
            npy_files = list(temp_dir.glob("*.npy"))

            # Fallback: CXAS sometimes writes next to the image instead of out_path
            if not npy_files:
                alt_npy = img_file.with_suffix(".npy")
                if alt_npy.exists():
                    # Move it into temp_dir so downstream logic works
                    target = temp_dir / alt_npy.name
                    alt_npy.replace(target)
                    npy_files = [target]

            if npy_files:
                npy_path = npy_files[0]
                full_masks = np.load(npy_path)
                
                all_organs_dict = {}
                
                for need_class, sub_classes in need_mask_table.items():
                    if len(sub_classes) == 0:
                        continue
                    
                    selected_indices = []
                    for each_class in sub_classes:
                        for label_name in label_mapper_image_name[each_class]:
                            idx = label_to_idx.get(label_name)
                            if idx is not None:
                                selected_indices.append(idx)
                    
                    if not selected_indices:
                        continue
                    
                    selected = full_masks[selected_indices]
                    selected_t = torch.from_numpy(selected).unsqueeze(1).float()
                    resized = F.interpolate(selected_t, size=(224, 224), mode="nearest").squeeze(1)
                    resized = (resized > 0.5).float()
                    all_organs_dict[need_class] = resized
                
                save_as_npz_unified(str(npz_path), all_organs_dict)
        
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

def update_config(args):
    #### IU xray
    config.iu_xray_caption = join(args.iu_xray_path, "annotation.json")
    config.iu_xray_image = join(args.iu_xray_path, "images")
    config.iu_xray_saved_path = join(args.data_path,"iu_xray_segmentation")
    #### Mimic-cxr
    config.mimic_cxr_caption = join(args.mimic_cxr_path, "annotation.json")
    config.mimic_cxr_image = join(args.mimic_cxr_path, "images")
    config.mimic_cxr_saved_path = join(args.data_path,"mimic_cxr_segmentation")

def main(args):
    if 'HOME' not in os.environ:
        os.environ['HOME'] = os.path.expanduser('~')
    torch.serialization.add_safe_globals([argparse.Namespace])
    cxas_model = create_cxas_model()
    update_config(args)
    
    # Process a single folder if specified
    if args.input_folder:
        print(f"Processing single folder: {args.input_folder}")
        segmentation_cxas_mimic_cxr_folder(cxas_model, args.input_folder, args)
    elif args.dataset_name == "iu_xray":
        caption_data = get_caption(config.iu_xray_caption)
        segmentation_cxas_iu_xray(cxas_model, caption_data, args)
        saved_mask_pickle_iu_xray()
    else:
        caption_data = get_caption(config.mimic_cxr_caption)
        segmentation_cxas_mimic_cxr(cxas_model, caption_data, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    project_directory = Path(__file__).parent.parent
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--keep_npy', action='store_true',
                        help='keep intermediate {0,1}.npy masks after creating unified_masks.npz')
    parser.add_argument('--input_folder', type=str, default=None,
                        help='(MIMIC only) process a single folder (e.g., /path/to/images/files/p10). '
                             'If specified, only this folder is processed instead of the entire dataset.')
    parser.add_argument('--data_path', type=str, default=join(project_directory, "data"),
                        help='the dataset path of total data')
    parser.add_argument('--iu_xray_path', type=str, default=join(project_directory, "data", "iu_xray"),
                        help='the dataset path of iu_xray.')
    parser.add_argument('--mimic_cxr_path', type=str, default=join(project_directory, "data", "mimic_cxr"),
                        help='the dataset path of mimic_xray.')
    args = parser.parse_args()
    main(args)
