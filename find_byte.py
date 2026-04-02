import os

def find_byte_9d(root_dir):
    for root, dirs, files in os.walk(root_dir):
        if ".git" in dirs:
            dirs.remove(".git")
        for file in files:
            path = os.path.join(root, file)
            try:
                with open(path, 'rb') as f:
                    content = f.read()
                    if b'\x9d' in content:
                        print(f"FOUND 0x9d in {path} at position {content.find(b'x9d')}")
                        # Print surrounding 20 bytes for context
                        pos = content.find(b'\x9d')
                        start = max(0, pos - 10)
                        end = min(len(content), pos + 10)
                        print(f"Context: {content[start:end]}")
            except Exception as e:
                pass

if __name__ == "__main__":
    find_byte_9d(".")
