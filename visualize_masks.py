"""
Visualization utility with selective organ loading.
Generates mask overlays on-demand without pre-saving JPGs.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_mask import load_from_npz_unified, need_mask_table, join

ORGAN_COLORS = {
    "bone": (1.0, 0.0, 0.0),
    "lung": (0.0, 0.0, 1.0),
    "heart": (0.0, 1.0, 0.0),
    "mediastinum": (1.0, 1.0, 0.0),
}

def load_original_image(image_path, size=(224, 224)):
    """Load and resize original image to match mask size"""
    img = Image.open(image_path).convert('L')
    img = img.resize(size, Image.BILINEAR)
    return np.array(img) / 255.0

def visualize_selected_organs(image_path, mask_npz_path, organs_to_show, output_dir=None):
    """
    Visualize selected organs with overlay on original image.
    
    Args:
        image_path: path to original X-ray image
        mask_npz_path: path to unified_masks.npz file
        organs_to_show: list of organ names to visualize 
                       (e.g., ['bone', 'lung'] or ['all'] for all)
        output_dir: directory to save visualizations (optional)
    """
    # Load image
    img = load_original_image(image_path)
    
    # Load only selected organs
    if organs_to_show == ['all']:
        masks_dict = load_from_npz_unified(mask_npz_path)  # Load all
    else:
        masks_dict = load_from_npz_unified(mask_npz_path, organs=organs_to_show)
    
    if not masks_dict:
        print(f"❌ No masks found for: {organs_to_show}")
        return
    
    # Create figure with subplots (1 per organ + original)
    num_organs = len(masks_dict)
    fig, axes = plt.subplots(1, num_organs + 1, figsize=(5 * (num_organs + 1), 5))
    
    # Ensure axes is always iterable (single subplot returns scalar, multiple return array)
    if num_organs + 1 == 1:
        axes = [axes]
    elif not hasattr(axes, '__iter__'):
        axes = [axes]
    
    # Show original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show each selected organ
    for idx, (organ_name, mask) in enumerate(masks_dict.items()):
        mask_np = mask.numpy()  # [C, 224, 224]
        union_mask = mask_np.max(axis=0)  # [224, 224]
        
        axes[idx + 1].imshow(img, cmap='gray')
        axes[idx + 1].imshow(union_mask, cmap='RdYlGn', alpha=0.4)
        axes[idx + 1].set_title(f'{organ_name} ({mask_np.shape[0]} channels)')
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        organs_str = '_'.join(organs_to_show) if organs_to_show != ['all'] else 'all'
        output_path = join(output_dir, f"overlay_{organs_str}.png")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize selected organ masks')
    parser.add_argument('--patient_id', type=str, required=True,
                        help='Patient folder ID')
    parser.add_argument('--organs', type=str, default='all',
                        help='Organs to visualize (comma-separated: bone,lung,heart or "all")')
    parser.add_argument('--dataset_name', type=str, default='iu_xray',
                        choices=['iu_xray', 'mimic_cxr'])
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--view_idx', type=int, default=0,
                        help='View index (0=frontal, 1=lateral for iu_xray)')
    parser.add_argument('--iu_xray_path', type=str, default='data/iu_xray')
    parser.add_argument('--output_head', type=str, default=None,
                        help='Optional output head folder (default: *_segmentation_virtualize)')
    
    args = parser.parse_args()

    # Resolve paths robustly: if data paths are relative, anchor them at repo root
    repo_root = Path(__file__).parent.parent
    data_root = Path(args.data_path)
    if not data_root.is_absolute():
        data_root = repo_root / data_root
    iu_images_root = Path(args.iu_xray_path)
    if not iu_images_root.is_absolute():
        iu_images_root = repo_root / iu_images_root
    
    # Parse organs to show
    if args.organs.lower() == 'all':
        organs_to_show = ['all']
    else:
        organs_to_show = [o.strip() for o in args.organs.split(',')]
    
    # Setup paths
    if args.dataset_name == "iu_xray":
        segmentation_path = str(data_root / "iu_xray_segmentation")
        patient_folder = join(segmentation_path, args.patient_id)
        # Try both JPG and PNG for IU-Xray
        iu_img_dir = iu_images_root / "images" / args.patient_id
        jpg_candidate = iu_img_dir / f"{args.view_idx}.jpg"
        png_candidate = iu_img_dir / f"{args.view_idx}.png"
        if jpg_candidate.exists():
            image_path = str(jpg_candidate)
        elif png_candidate.exists():
            image_path = str(png_candidate)
        else:
            # Fallback: search any file matching the view index
            matches = list(iu_img_dir.glob(f"{args.view_idx}.*"))
            image_path = str(matches[0]) if matches else str(jpg_candidate)
        # IU-Xray uses view_mask subdirectory
        mask_dir = join(patient_folder, f"{args.view_idx}_mask")
    else:
        # MIMIC-CXR: patient_id is relative path like files/p10/p10xxx/s50xxx
        segmentation_path = str(data_root / "mimic_cxr_segmentation")
        patient_folder = join(segmentation_path, args.patient_id)
        image_path = str(data_root / "mimic_cxr" / "images" / args.patient_id)
        # Find image file (could be .jpg, .png, .dcm converted to jpg)
        img_dir = Path(image_path)
        if img_dir.is_dir():
            candidates = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            image_path = str(candidates[0]) if candidates else str(img_dir / "image.jpg")
        # MIMIC saves directly in study folder (no 0_mask subdirectory)
        mask_dir = patient_folder
    
    mask_npz_path = join(mask_dir, "unified_masks.npz")

    # Derive virtualize output root mirroring the original segmentation tree
    if args.output_head:
        output_head = args.output_head
    else:
        output_head = "iu_xray_segmentation_virtualize" if args.dataset_name == "iu_xray" else "mimic_cxr_segmentation_virtualize"
    virtualize_root = str(data_root / output_head)
    virtualize_patient_folder = join(virtualize_root, args.patient_id)
    # IU uses view_mask subfolder, MIMIC saves flat
    if args.dataset_name == "iu_xray":
        virtualize_mask_dir = join(virtualize_patient_folder, f"{args.view_idx}_mask")
    else:
        virtualize_mask_dir = virtualize_patient_folder
    
    # Validate
    if not os.path.exists(mask_npz_path):
        print(f"❌ Mask file not found: {mask_npz_path}")
        return
    
    if not os.path.exists(image_path):
        # Helpful debug: show available files in the expected folder
        expected_dir = Path(image_path).parent
        available = []
        if expected_dir.exists():
            available = [p.name for p in expected_dir.glob("*")]
        print(f"❌ Image not found: {image_path}")
        if available:
            print(f"   Available in folder: {available}")
        return
    
    print(f"📊 Visualizing patient {args.patient_id}, organs: {organs_to_show}")
    visualize_selected_organs(image_path, mask_npz_path, organs_to_show, output_dir=virtualize_mask_dir)

if __name__ == "__main__":
    main()
