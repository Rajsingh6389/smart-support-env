import os
import re

def find_non_ascii(root_dir):
    pattern = re.compile(r'[^\x00-\x7f]')
    for root, dirs, files in os.walk(root_dir):
        if ".git" in dirs:
            dirs.remove(".git")
        if ".venv" in dirs:
            dirs.remove(".venv")
            
        for file in files:
            if file.endswith(('.py', '.yaml', '.toml', '.md', 'Dockerfile', '.sh', '.gitignore')):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        matches = pattern.findall(content)
                        if matches:
                            print(f"Found non-ASCII in {path}: {set(matches)}")
                            # Replace with ASCII equivalent if possible, or space
                            # Common ones: emojis, curly quotes, dashes
                            content = content.replace('-', '-') # em dash
                            content = content.replace('-', '-') # en dash
                            content = content.replace(''', "'").replace(''', "'")
                            content = content.replace('"', '"').replace('"', '"')
                            # Generic strip remaining non-ascii
                            content = "".join([c if ord(c) < 128 else " " for c in content])
                            
                            with open(path, 'w', encoding='utf-8') as f_out:
                                f_out.write(content)
                            print(f"Cleaned {path}")
                except Exception as e:
                    print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    find_non_ascii(".")
