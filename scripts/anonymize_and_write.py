import os
import shutil
import re

# Configuration
PROJECT_ROOT = os.getcwd()
SOURCE_SRC = os.path.join(PROJECT_ROOT, "src")
TARGET_ROOT = os.path.join(PROJECT_ROOT, "EigenCache")
TARGET_SRC = os.path.join(TARGET_ROOT, "src")

# Sensitive word replacement table
REPLACEMENTS = {
    "EigenCache-Project": "EigenCache-Project",
    "anonymous": "anonymous",
    "/home/anonymous": "/home/anonymous",
    "/home/anonymous/project": "/home/anonymous/project",
    "/data/models": "/data/models",
    "model-provider": "model-provider", # Optional: if you need to hide the model source
}

# Files/folders to ignore
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
    # 1. Replace words from the dictionary
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)
    
    # 2. Handle absolute paths (Windows & Linux)
    # Match C:\Users\...\Desktop...
    content = re.sub(r"C:[\\/]Users[\\/][^\\/]+[\\/].*", "/home/anonymous/project", content)
    # Match /home/username/...
    content = re.sub(r"/home/[^/]+/", "/home/anonymous/", content)
    
    return content

def copy_and_anonymize(src_path, dst_path):
    # Ensure target directory exists
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    try:
        # Read
        with open(src_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Anonymize
        new_content = anonymize_content(content)
        
        # Write
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
    except UnicodeDecodeError:
        # If binary file (e.g., image), copy directly
        shutil.copy2(src_path, dst_path)

def process_directory(source, target):
    for root, dirs, files in os.walk(source):
        # Filter ignored folders
        dirs[:] = [d for d in dirs if d not in IGNORE_PATTERNS and not d.startswith(".")]
        
        for file in files:
            if any(file.endswith(ext.replace("*", "")) for ext in IGNORE_PATTERNS):
                continue
                
            src_file = os.path.join(root, file)
            # Calculate relative path
            rel_path = os.path.relpath(src_file, source)
            dst_file = os.path.join(target, rel_path)
            
            copy_and_anonymize(src_file, dst_file)

def main():
    print(f"Starting anonymization from {PROJECT_ROOT} to {TARGET_ROOT}...")

    # 1. Clean up old directory
    if os.path.exists(TARGET_ROOT):
        shutil.rmtree(TARGET_ROOT)
    os.makedirs(TARGET_ROOT)

    # 2. Process src directory
    print("Processing src directory...")
    process_directory(SOURCE_SRC, TARGET_SRC)

    # 3. Process key files in root directory
    root_files = ["README.md", "pyproject.toml", "requirements.txt", "LICENSE"]
    for file in root_files:
        src_file = os.path.join(PROJECT_ROOT, file)
        if os.path.exists(src_file):
            print(f"Processing {file}...")
            copy_and_anonymize(src_file, os.path.join(TARGET_ROOT, file))

    # 4. Process scripts directory (optional, usually contains run logic)
    print("Processing scripts directory...")
    process_directory(os.path.join(PROJECT_ROOT, "scripts"), os.path.join(TARGET_ROOT, "scripts"))

    # 5. Generate anonymous README
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
