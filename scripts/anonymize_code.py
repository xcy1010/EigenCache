import os
import shutil
import re

# 配置
SOURCE_DIR = "src"
TARGET_DIR = "EigenCache"
SENSITIVE_PATTERNS = [
    r"anonymous",  # 用户名
    r"EigenCache-Project",  # 项目名
    r"/home/anonymous/project
    r"/home/anonymous/]+",  # Linux 绝对路径
    r"/data/models",  # AutoDL 路径
    r"model-provider", # 机构名 (可选，视比赛要求而定，这里先保留或替换为 generic-org)
]

REPLACEMENTS = {
    "EigenCache-Project": "EigenCache-Project",
    "flux": "eigencache_core", # 将 flux 包重命名为 eigencache_core 以隐藏来源
    "anonymous": "anonymous_user",
    "/data/models": "/tmp/data",
}

def anonymize_content(content):
    for pattern in SENSITIVE_PATTERNS:
        # 简单的字符串替换，对于正则需要更复杂的处理，这里主要处理明确的字符串
        pass
    
    # 使用替换字典
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)
        
    # 处理绝对路径正则
    content = re.sub(r"C:[\\/]Users[\\/][^\\/]+", "/home/anonymous/home/[^/]+", "/home/anonymous/文件夹中的敏感词
        for old, new in REPLACEMENTS.items():
            if old in item:
                d = os.path.join(dst, item.replace(old, new))
                
        if os.path.isdir(s):
            # 忽略 __pycache__
            if item == "__pycache__" or item.startswith("."):
                continue
            process_directory(s, d)
        else:
            # 只处理文本文件
            if s.endswith((".py", ".md", ".txt", ".sh", ".toml")):
                try:
                    with open(s, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    new_content = anonymize_content(content)
                    
                    with open(d, "w", encoding="utf-8") as f:
                        f.write(new_content)
                except Exception as e:
                    print(f"Skipping binary or unreadable file: {s}")
                    shutil.copy2(s, d)
            else:
                shutil.copy2(s, d)

def main():
    # 1. 清理旧的 EigenCache 文件夹
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    
    # 2. 复制并脱敏 src 目录
    print(f"Processing {SOURCE_DIR} -> {TARGET_DIR}/src...")
    process_directory(SOURCE_DIR, os.path.join(TARGET_DIR, "src"))
    
    # 3. 复制并脱敏 README 和 requirements (如果有)
    for file in ["README.md", "pyproject.toml", "requirements.txt"]:
        if os.path.exists(file):
            print(f"Processing {file}...")
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            new_content = anonymize_content(content)
            with open(os.path.join(TARGET_DIR, file), "w", encoding="utf-8") as f:
                f.write(new_content)

    # 4. 创建一个简单的 setup 说明
    with open(os.path.join(TARGET_DIR, "ANONYMOUS_SETUP.md"), "w", encoding="utf-8") as f:
        f.write("# Anonymous Submission\n\nThis code has been anonymized for peer review.\n\n## Usage\n\nSee `src/sample.py` for inference logic.")

    print("Done! Anonymized code is in 'EigenCache' directory.")

if __name__ == "__main__":
    main()
