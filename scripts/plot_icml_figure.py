import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from PIL import Image
import numpy as np

# ================= 配置区域 =================

# 1. 根目录 (已根据您的要求修改)
ROOT_DIR = r"C:\Users\xcy\Desktop\ICML\3\HiCache-Flux-Pub\results_mini"

# 2. 定义要对比的方法 (显示名称, 文件夹相对路径)
# 注意：请根据实际生成的文件夹名称进行微调，特别是带有参数后缀的文件夹
METHODS = [
    ("Original (GT)", "original/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("ToCa",          "toca/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("Taylor",        "taylor/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    ("HiCache (Ours)","hicache/mn_flux-dev_i_6_o_2_s_50_hs_0.7"),
    # 如果有 EigenCache，请取消注释并确认路径
    # ("EigenCache",    "taylor/mn_flux-dev_i_6_o_2_s_50_hs_0.7_cm_eigencache_sch_fixed"), 
]

# 3. 选择要展示的图片索引 (对应 prompt_icml_mini_10.txt 的行号，从0开始)
# 建议选择：0(文字), 1(人像), 2(光影), 3(艺术), 5(微距)
SELECTED_INDICES = [0, 1, 2, 3] 

# 4. 定义每张图的“找茬”放大区域 (x, y, width, height)
# 坐标系：原图左上角为 (0,0)。Flux 生成图通常是 1024x1024。
# 如果不设置，默认放大中心区域。
CROP_CONFIG = {
    0: (400, 400, 200, 200),  # 示例：针对 Prompt 0 (文字) 关注文字区域
    1: (450, 300, 150, 150),  # 示例：针对 Prompt 1 (人像) 关注眼睛/皮肤
    2: (300, 500, 200, 200),  # 示例：针对 Prompt 2 (棋盘) 关注折射
    3: (500, 500, 200, 200),  # 示例：针对 Prompt 3 (油画) 关注笔触
}

# ================= 绘图逻辑 =================

def crop_image(img, coords):
    x, y, w, h = coords
    return img.crop((x, y, x+w, y+h))

def find_image_path(base_path, idx):
    """尝试寻找 jpg 或 png"""
    # 尝试常见命名格式
    names = [f"img_{idx}.jpg", f"img_{idx}.png", f"{idx:05d}.png", f"{idx:05d}.jpg"]
    
    # 搜索目录（处理可能存在的子目录结构）
    if not os.path.exists(base_path):
        print(f"Warning: Path not found: {base_path}")
        return None
        
    for root, dirs, files in os.walk(base_path):
        for name in names:
            if name in files:
                return os.path.join(root, name)
    return None

def plot_comparison():
    # 设置字体
    plt.rcParams['font.family'] = 'DejaVu Sans' # 论文常用无衬线字体
    plt.rcParams['font.size'] = 12

    num_rows = len(SELECTED_INDICES)
    num_cols = len(METHODS)
    
    # 创建画布：每张图宽 3 英寸，高 3 英寸
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.5 * num_cols, 3.5 * num_rows), 
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
    
    if num_rows == 1: axes = [axes] # 统一维度处理

    for row_idx, img_idx in enumerate(SELECTED_INDICES):
        # 获取该图片的裁剪坐标，默认中心裁剪
        crop_coords = CROP_CONFIG.get(img_idx, (412, 412, 200, 200))
        
        for col_idx, (method_name, rel_path) in enumerate(METHODS):
            ax = axes[row_idx][col_idx]
            
            full_path = os.path.join(ROOT_DIR, rel_path)
            img_path = find_image_path(full_path, img_idx)
            
            if img_path:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                
                # --- 绘制放大插图 (Inset) ---
                # 创建插图轴：位于右下角，大小为原图的 35%
                axins = inset_axes(ax, width="35%", height="35%", loc='lower right', borderpad=0.5)
                
                # 裁剪并显示
                zoomed_img = crop_image(img, crop_coords)
                axins.imshow(zoomed_img)
                
                # 设置插图边框颜色 (红色醒目)
                for spine in axins.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
                axins.set_xticks([])
                axins.set_yticks([])
                
                # --- 绘制原图上的定位框和连线 ---
                # mark_inset 会自动画出原图上的框和两条连线
                # loc1, loc2 控制连线的角点 (1=右上, 2=左上, 3=左下, 4=右下)
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="red", lw=2)
                
            else:
                ax.text(0.5, 0.5, "Image Not Found", ha='center', va='center', color='red')
                ax.set_facecolor('#f0f0f0')

            # --- 样式调整 ---
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 仅在第一行显示方法名称
            if row_idx == 0:
                ax.set_title(method_name, fontweight='bold', pad=10, fontsize=14)
            
            # 仅在第一列显示 Prompt 编号 (可选)
            if col_idx == 0:
                ax.set_ylabel(f"Prompt {img_idx}", fontweight='bold', fontsize=12)

    # 保存结果
    output_file = "icml_comparison_figure.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Successfully saved comparison figure to {output_file}")
    
    # 同时保存 PNG 用于预览
    plt.savefig(output_file.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_comparison()
