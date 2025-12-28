$pythonScript = @"
import os
import shutil
import datetime


def is_ascii(text: str) -> bool:
    """Check whether a string contains ASCII characters only"""
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def main():
    # Read user input
    src_dir = input("Enter source folder path: ").strip()
    dst_root = input("Enter output folder path: ").strip()

    if not os.path.isdir(src_dir):
        print("Source folder does not exist.")
        input("Press any key to exit...")
        return

    if not os.path.isdir(dst_root):
        print("Output folder does not exist.")
        input("Press any key to exit...")
        return

    # Prepare destination folder name
    src_dir = os.path.abspath(src_dir)
    dst_root = os.path.abspath(dst_root)

    src_name = os.path.basename(src_dir.rstrip("\\/"))
    today = datetime.datetime.now().strftime("%Y%m%d")
    dst_dir = os.path.join(dst_root, f"{src_name}_{today}")

    os.makedirs(dst_dir, exist_ok=True)

    counter = 1

    # Walk through source directory
    for root, _, files in os.walk(src_dir):
        for file_name in files:
            src_file = os.path.join(root, file_name)

            # Build relative path
            rel_path = os.path.relpath(src_file, src_dir)
            rel_no_ext, ext = os.path.splitext(rel_path)

            # Replace path separators with underscore
            safe_name = rel_no_ext.replace("\\", "_").replace("/", "_")

            if is_ascii(file_name):
                new_name = safe_name + ext
            else:
                new_name = f"{safe_name.rsplit('_', 1)[0]}_{counter:04d}{ext}"
                counter += 1

            dst_file = os.path.join(dst_dir, new_name)

            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)

            shutil.copy2(src_file, dst_file)

    print("Copy completed successfully.")
    input("Press any key to exit...")


if __name__ == "__main__":
    main()

"@
$scriptPath = "$env:TEMP\temp_script.py"
Set-Content -Path $scriptPath -Value $pythonScript
python $scriptPath
Remove-Item $scriptPath  # 清理临时文件