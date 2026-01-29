import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ================= Configuration Area =================

# 1. Root directory
ROOT_DIR = r"/home/anonymous/project"

# 2. Define methods to compare (Display Name, Relative Folder Path)
METHODS = [
    ("Original (GT)", "original/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("ToCa",          "toca/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("Taylor",        "taylor/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("HiCache (Ours)","hicache/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    # ("EigenCache",    "taylor/mn_flux-dev_i_6_o_2_s_50_hs_0.7_cm_eigencache_sch_fixed"),
]

# 3. Select image indices to display
SELECTED_INDICES = [0, 1, 2, 3]

# 4. Whether to show difference map (Heatmap)
SHOW_DIFF = True

# ===========================================

def load_image(path):
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path).convert("RGB"))

def find_image_path(base_path, idx):
    names = [f"img_{idx}.jpg", f"img_{idx}.png", f"{idx:05d}.png", f"{idx:05d}.jpg"]
    if not os.path.exists(base_path):
        return None
    for root, dirs, files in os.walk(base_path):
        for name in names:
            if name in files:
                return os.path.join(root, name)
    return None

def plot_comparison():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10

    num_samples = len(SELECTED_INDICES)
    num_methods = len(METHODS)
    
    # If showing difference map, double the rows (one for original, one for difference)
    rows_per_sample = 2 if SHOW_DIFF else 1
    total_rows = num_samples * rows_per_sample
    
    fig, axes = plt.subplots(total_rows, num_methods, figsize=(3 * num_methods, 3 * total_rows),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.1})
    
    if total_rows == 1: axes = [axes]
    if num_methods == 1: axes = [[ax] for ax in axes]

    gt_images = {} # Cache GT images for difference calculation

    for i, img_idx in enumerate(SELECTED_INDICES):
        row_base = i * rows_per_sample
        
        # Load GT image first
        gt_path = find_image_path(os.path.join(ROOT_DIR, METHODS[0][1]), img_idx)
        if gt_path:
            gt_images[img_idx] = load_image(gt_path)
        else:
            print(f"Warning: GT image not found for index {img_idx}")
            gt_images[img_idx] = None

        for j, (method_name, rel_path) in enumerate(METHODS):
            ax_img = axes[row_base][j]
            
            full_path = os.path.join(ROOT_DIR, rel_path)
            img_path = find_image_path(full_path, img_idx)
            
            img_data = load_image(img_path)
            
            if img_data is not None:
                # 1. Show original image
                ax_img.imshow(img_data)
                
                # Calculate metrics (relative to GT)
                metric_text = ""
                if j > 0 and gt_images[img_idx] is not None and img_data.shape == gt_images[img_idx].shape:
                    try:
                        p_val = psnr(gt_images[img_idx], img_data, data_range=255)
                        s_val = ssim(gt_images[img_idx], img_data, channel_axis=2, data_range=255)
                        metric_text = f"\nPSNR: {p_val:.2f} | SSIM: {s_val:.3f}"
                    except Exception as e:
                        print(f"Metric error: {e}")

                # Set title (only for the first row)
                if i == 0:
                    ax_img.set_title(method_name, fontweight='bold', fontsize=12)
                
                # Set left label (only for the first column)
                if j == 0:
                    ax_img.set_ylabel(f"Prompt {img_idx}", fontweight='bold', fontsize=10)

                # Add metric text below image
                if metric_text:
                    ax_img.set_xlabel(metric_text, fontsize=8)

                # 2. Show difference map (if enabled)
                if SHOW_DIFF:
                    ax_diff = axes[row_base + 1][j]
                    if j == 0:
                        # GT difference map is black, or show "Reference"
                        ax_diff.text(0.5, 0.5, "Reference", ha='center', va='center')
                        ax_diff.set_facecolor('#f0f0f0')
                        ax_diff.set_ylabel("Difference", fontsize=9, fontstyle='italic')
                    elif gt_images[img_idx] is not None and img_data.shape == gt_images[img_idx].shape:
                        # Calculate absolute difference
                        diff = np.abs(gt_images[img_idx].astype(np.float32) - img_data.astype(np.float32))
                        # Convert to grayscale and enhance contrast (x5) for visibility
                        diff_gray = np.mean(diff, axis=2) * 5.0
                        diff_gray = np.clip(diff_gray, 0, 255).astype(np.uint8)
                        
                        # Show using heatmap (magma: black->red->yellow)
                        ax_diff.imshow(diff_gray, cmap='magma')
                    else:
                        ax_diff.text(0.5, 0.5, "N/A", ha='center', va='center')
                    
                    ax_diff.set_xticks([])
                    ax_diff.set_yticks([])

            else:
                ax_img.text(0.5, 0.5, "Not Found", ha='center', va='center', color='red')
            
            ax_img.set_xticks([])
            ax_img.set_yticks([])

    output_file = "video_comparison_figure.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    plot_comparison()
