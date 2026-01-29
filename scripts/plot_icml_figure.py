import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from PIL import Image
import numpy as np

# ================= Configuration Area =================

# 1. Root directory (Modified according to your request)
ROOT_DIR = r"C:\Users\xcy\Desktop\ICML\3\HiCache-Flux-Pub\results_mini"

# 2. Define methods to compare (Display Name, Relative Folder Path)
# Note: Please fine-tune according to the actual generated folder names, especially folders with parameter suffixes
METHODS = [
    ("Original (GT)", "original/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("ToCa",          "toca/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("Taylor",        "taylor/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("HiCache (Ours)","hicache/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    # If EigenCache is available, uncomment and confirm path
    # ("EigenCache",    "taylor/mn_flux-dev_i_6_o_2_s_50_hs_0.7_cm_eigencache_sch_fixed"), 
]

# 3. Select image indices to display (Corresponding to line numbers in prompt_icml_mini_10.txt, starting from 0)
# Suggested selection: 0 (Text), 1 (Portrait), 2 (Lighting), 3 (Art), 5 (Macro)
SELECTED_INDICES = [0, 1, 2, 3] 

# 4. Define "Spot the Difference" zoom area for each image (x, y, width, height)
# Coordinate system: Top-left of the original image is (0,0). Flux generated images are usually 1024x1024.
# If not set, defaults to zooming in the center.
CROP_CONFIG = {
    0: (400, 400, 200, 200),  # Example: For Prompt 0 (Text), focus on text area
    1: (450, 300, 150, 150),  # Example: For Prompt 1 (Portrait), focus on eyes/skin
    2: (300, 500, 200, 200),  # Example: For Prompt 2 (Chessboard), focus on refraction
    3: (500, 500, 200, 200),  # Example: For Prompt 3 (Oil Painting), focus on brushstrokes
}

# ================= Plotting Logic =================

def crop_image(img, coords):
    x, y, w, h = coords
    return img.crop((x, y, x+w, y+h))

def find_image_path(base_path, idx):
    """Try to find jpg or png"""
    # Try common naming formats
    names = [f"img_{idx}.jpg", f"img_{idx}.png", f"{idx:05d}.png", f"{idx:05d}.jpg"]
    
    # Search directory (handle possible subdirectory structures)
    if not os.path.exists(base_path):
        print(f"Warning: Path not found: {base_path}")
        return None
        
    for root, dirs, files in os.walk(base_path):
        for name in names:
            if name in files:
                return os.path.join(root, name)
    return None

def plot_comparison():
    # Set font
    plt.rcParams['font.family'] = 'DejaVu Sans' # Common sans-serif font for papers
    plt.rcParams['font.size'] = 12

    num_rows = len(SELECTED_INDICES)
    num_cols = len(METHODS)
    
    # Create canvas: Each image is 3 inches wide and 3 inches high
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5 * num_cols, 3.5 * num_rows), 
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    
    if num_rows == 1: axes = [axes] # Unify dimension handling

    for row_idx, img_idx in enumerate(SELECTED_INDICES):
        # Get crop coordinates for this image, default to center crop
        crop_coords = CROP_CONFIG.get(img_idx, (412, 412, 200, 200))
        
        for col_idx, (method_name, rel_path) in enumerate(METHODS):
            ax = axes[row_idx][col_idx]
            
            full_path = os.path.join(ROOT_DIR, rel_path)
            img_path = find_image_path(full_path, img_idx)
            
            if img_path:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                
                # --- Draw Zoom-in Inset ---
                # Create inset axes: located at bottom right, size is 35% of original image
                axins = inset_axes(ax, width="35%", height="35%", loc='lower right', borderpad=0.5)
                
                # Crop and display
                zoomed_img = crop_image(img, crop_coords)
                axins.imshow(zoomed_img)
                
                # Set inset border color (Red for visibility)
                for spine in axins.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
                axins.set_xticks([])
                axins.set_yticks([])
                
                # --- Draw positioning box and connecting lines on original image ---
                # mark_inset automatically draws the box on the original image and two connecting lines
                # loc1, loc2 control the corners of the connecting lines (1=top-right, 2=top-left, 3=bottom-left, 4=bottom-right)
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red", lw=2)
                
            else:
                ax.text(0.5, 0.5, "Image Not Found", ha='center', va='center', color='red')
                ax.set_facecolor('#f0f0f0')

            # --- Style Adjustments ---
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Display method name only in the first row
            if row_idx == 0:
                ax.set_title(method_name, fontweight='bold', pad=10, fontsize=14)
            
            # Display Prompt number only in the first column (Optional)
            if col_idx == 0:
                ax.set_ylabel(f"Prompt {img_idx}", fontweight='bold', fontsize=12)

    # Save result
    output_file = "icml_comparison_figure.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Successfully saved comparison figure to {output_file}")
    
    # Also save PNG for preview
    plt.savefig(output_file.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_comparison()
