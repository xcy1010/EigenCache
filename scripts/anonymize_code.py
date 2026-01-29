import os
import shutil
import re

# Configuration
SOURCE_DIR = "src"
TARGET_DIR = "EigenCache"
SENSITIVE_PATTERNS = [
    r"anonymous",  # Username
    r"EigenCache-Project",  # Project name
    r"/home/anonymous/project", # Windows absolute path
    r"/home/anonymous/]+",  # Linux absolute path
    r"/data/models",  # AutoDL path
    r"model-provider", # Organization name (optional, depends on competition rules, keep or replace with generic-org)
]

REPLACEMENTS = {
    "EigenCache-Project": "EigenCache-Project",
    "flux": "eigencache_core", # Rename flux package to eigencache_core to hide origin
    "anonymous": "anonymous_user",
    "/data/models": "/tmp/data",
}

def anonymize_content(content):
    for pattern in SENSITIVE_PATTERNS:
        # Simple string replacement, regex needs more complex handling, here mainly handles explicit strings
        pass
    
    # Use replacement dictionary
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)
        
    # Handle absolute path regex
    content = re.sub(r"C:[\\/]Users[\\/][^\\/]+", "/home/anonymous", content)
    content = re.sub(r"/home/[^/]+", "/home/anonymous", content)
    
    return content

def process_directory(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
        
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        
        # Rename sensitive words in file/folder names
        for old, new in REPLACEMENTS.items():
            if old in item:
                d = os.path.join(dst, item.replace(old, new))
                
        if os.path.isdir(s):
            # Ignore __pycache__
            if item == "__pycache__" or item.startswith("."):
                continue
            process_directory(s, d)
        else:
            # Only process text files
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
    # 1. Clean up old EigenCache folder
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    
    # 2. Copy and anonymize src directory
    print(f"Processing {SOURCE_DIR} -> {TARGET_DIR}/src...")
    process_directory(SOURCE_DIR, os.path.join(TARGET_DIR, "src"))
    
    # 3. Copy and anonymize README and requirements (if any)
    for file in ["README.md", "pyproject.toml", "requirements.txt"]:
        if os.path.exists(file):
            print(f"Processing {file}...")
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            new_content = anonymize_content(content)
            with open(os.path.join(TARGET_DIR, file), "w", encoding="utf-8") as f:
                f.write(new_content)

    # 4. Create a simple setup instruction
    with open(os.path.join(TARGET_DIR, "ANONYMOUS_SETUP.md"), "w", encoding="utf-8") as f:
        f.write("# Anonymous Submission\n\nThis code has been anonymized for peer review.\n\n## Usage\n\nSee `src/sample.py` for inference logic.")

    print("Done! Anonymized code is in 'EigenCache' directory.")

if __name__ == "__main__":
    main()
