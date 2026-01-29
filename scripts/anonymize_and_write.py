import os
import shutil
import re

# 配置
PROJECT_ROOT = os.getcwd()
SOURCE_SRC = os.path.join(PROJECT_ROOT, "src")
TARGET_ROOT = os.path.join(PROJECT_ROOT, "EigenCache")
TARGET_SRC = os.path.join(TARGET_ROOT, "src")

# 敏感词替换表
REPLACEMENTS = {
    "EigenCache-Project": "EigenCache-Project",
    "anonymous": "anonymous",
    "/home/anonymous/home/anonymous",
    "/home/anonymous/project
    "/data/models": "/data/models",
    "model-provider": "model-provider", # 可选：如果需要隐藏模型来源
}

# 需要忽略的文件/文件夹
IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".git",
    ".idea",
    "logs",
    "results",
    "wandb"
]

def anonymize_content(content):
    # 1. 替换字典中的词
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)
    
    # 2. 处理绝对路径 (Windows & Linux)
    # 匹配 /home/anonymous/project
    content = re.sub(r"C:[\\/]Users[\\/][^\\/]+[\\/].*", "/home/anonymous/project", content)
    # 匹配 /home/anonymous/...
    content = re.sub(r"/home/anonymous/]+/", "/home/anonymous/", content)
    
    return content

def copy_and_anonymize(src_path, dst_path):
    # 确保目标目录存在
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    try:
        # 读取
        with open(src_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 脱敏
        new_content = anonymize_content(content)
        
        # 写入
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
    except UnicodeDecodeError:
        # 如果是二进制文件（如图片），直接复制
        shutil.copy2(src_path, dst_path)

def process_directory(source, target):
    for root, dirs, files in os.walk(source):
        # 过滤忽略的文件夹
        dirs[:] = [d for d in dirs if d not in IGNORE_PATTERNS and not d.startswith(".")]
        
        for file in files:
            if any(file.endswith(ext.replace("*", "")) for ext in IGNORE_PATTERNS):
                continue
                
            src_file = os.path.join(root, file)
            # 计算相对路径
            rel_path = os.path.relpath(src_file, source)
            dst_file = os.path.join(target, rel_path)
            
            copy_and_anonymize(src_file, dst_file)

def main():
    print(f"Starting anonymization from {PROJECT_ROOT} to {TARGET_ROOT}...")

    # 1. 清理旧目录
    if os.path.exists(TARGET_ROOT):
        shutil.rmtree(TARGET_ROOT)
    os.makedirs(TARGET_ROOT)

    # 2. 处理 src 目录
    print("Processing src directory...")
    process_directory(SOURCE_SRC, TARGET_SRC)

    # 3. 处理根目录下的关键文件
    root_files = ["README.md", "pyproject.toml", "requirements.txt", "LICENSE"]
    for file in root_files:
        src_file = os.path.join(PROJECT_ROOT, file)
        if os.path.exists(src_file):
            print(f"Processing {file}...")
            copy_and_anonymize(src_file, os.path.join(TARGET_ROOT, file))

    # 4. 处理 scripts 目录 (可选，通常包含运行逻辑)
    print("Processing scripts directory...")
    process_directory(os.path.join(PROJECT_ROOT, "scripts"), os.path.join(TARGET_ROOT, "scripts"))

    # 5. 生成匿名说明
    with open(os.path.join(TARGET_ROOT, "ANONYMOUS_README.md"), "w", encoding="utf-8") as f:
        f.write("# EigenCache (Anonymous Submission)\n\n")
        f.write("This repository contains the source code for EigenCache.\n")
        f.write("All personal information and absolute paths have been anonymized.\n\n")
        f.write("## Structure\n")
        f.write("- `src/`: Core implementation\n")
        f.write("- `scripts/`: Helper scripts for running experiments\n")

    print("✅ Anonymization complete!")

if __name__ == "__main__":
    main()
