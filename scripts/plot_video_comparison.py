import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ================= 配置区域 =================

# 1. 根目录
ROOT_DIR = r"/home/anonymous/project

# 2. 定义要对比的方法 (显示名称, 文件夹相对路径)
METHODS = [
    ("Original (GT)", "original/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("ToCa",          "toca/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("Taylor",        "taylor/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("HiCache (Ours)","hicache/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    # ("EigenCache",    "taylor/mn_flux-dev_i_6_o_2_s_50_hs_0.7_cm_eigencache_sch_fixed"),
]

# 3. 选择要展示的图片索引
SELECTED_INDICES = [0, 1, 2, 3]

# 4. 是否显示差值图 (Heatmap)
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
    
    # 如果显示差值图，行数翻倍（原图一行，差值图一行）
    rows_per_sample = 2 if SHOW_DIFF else 1
    total_rows = num_samples * rows_per_sample
    
    fig, axes = plt.subplots(total_rows, num_methods, figsize=(3 * num_methods, 3 * total_rows),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.1})
    
    if total_rows == 1: axes = [axes]
    if num_methods == 1: axes = [[ax] for ax in axes]

    gt_images = {} # 缓存 GT 图片用于计算差值

    for i, img_idx in enumerate(SELECTED_INDICES):
        row_base = i * rows_per_sample
        
        # 先加载 GT 图片
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
                # 1. 显示原图
                ax_img.imshow(img_data)
                
                # 计算指标 (相对于 GT)
                metric_text = ""
                if j > 0 and gt_images[img_idx] is not None and img_data.shape == gt_images[img_idx].shape:
                    try:
                        p_val = psnr(gt_images[img_idx], img_data, data_range=255)
                        s_val = ssim(gt_images[img_idx], img_data, channel_axis=2, data_range=255)
                        metric_text = f"\nPSNR: {p_val:.2f} | SSIM: {s_val:.3f}"
                    except Exception as e:
                        print(f"Metric error: {e}")

                # 设置标题（仅第一行）
                if i == 0:
                    ax_img.set_title(method_name, fontweight='bold', fontsize=12)
                
                # 设置左侧标签（仅第一列）
                if j == 0:
                    ax_img.set_ylabel(f"Prompt {img_idx}", fontweight='bold', fontsize=10)

                # 在图片下方添加指标文字
                if metric_text:
                    ax_img.set_xlabel(metric_text, fontsize=8)

                # 2. 显示差值图 (如果有)
                if SHOW_DIFF:
                    ax_diff = axes[row_base + 1][j]
                    if j == 0:
                        # GT 的差值图是全黑，或者显示 "Reference"
                        ax_diff.text(0.5, 0.5, "Reference", ha='center', va='center')
                        ax_diff.set_facecolor('#f0f0f0')
                        ax_diff.set_ylabel("Difference", fontsize=9, fontstyle='italic')
                    elif gt_images[img_idx] is not None and img_data.shape == gt_images[img_idx].shape:
                        # 计算绝对差值
                        diff = np.abs(gt_images[img_idx].astype(np.float32) - img_data.astype(np.float32))
                        # 转为灰度并增强对比度 (x5) 以便观察
                        diff_gray = np.mean(diff, axis=2) * 5.0
                        diff_gray = np.clip(diff_gray, 0, 255).astype(np.uint8)
                        
                        # 使用热力图显示 (magma: 黑->红->黄)
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
