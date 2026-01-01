$pythonScript = @"
import os
import sys
import shutil
import datetime


def is_ascii(s):
    """Check if a string consists entirely of ASCII characters."""
    return all(ord(c) < 128 for c in s)


def get_mapped_name(name, mapping, counter_ref):
    """
    Returns the original name if ASCII.
    If non-ASCII, returns a 4-digit number based on a mapping dictionary.
    """
    if is_ascii(name):
        return name

    # If this specific non-ascii string hasn't been seen yet, assign a new ID
    if name not in mapping:
        mapping[name] = f"{counter_ref[0]:04d}"
        counter_ref[0] += 1

    return mapping[name]


def main():
    print("--- WAV File Processor ---")

    # 1. Get Inputs
    input_dir = input("1. Please enter the input folder path (e.g., d:/mp3): ").strip()
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        input("Press any key to exit...")
        sys.exit(1)

    output_root = input("2. Please enter the output folder path (e.g., f:/): ").strip()
    if not os.path.exists(output_root):
        # Try to create it if the root doesn't exist, though usually drives must exist
        try:
            os.makedirs(output_root, exist_ok=True)
        except OSError:
            print(f"Error: Output path '{output_root}' is invalid or not accessible.")
            input("Press any key to exit...")
            sys.exit(1)

    # 2. Create Destination Folder
    # Format: [SourceDirName]-YYYYMMDD
    source_folder_name = os.path.basename(os.path.normpath(input_dir))
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    dest_folder_name = f"{source_folder_name}-{date_str}"
    dest_path = os.path.join(output_root, dest_folder_name)

    try:
        os.makedirs(dest_path, exist_ok=True)
        print(f"Target folder created: {dest_path}")
    except OSError as e:
        print(f"Error creating destination folder: {e}")
        input("Press any key to exit...")
        sys.exit(1)

    # 3. Process Files
    print("Processing files...")

    # Dictionary to store mapping of Non-ASCII strings to "0001", "0002", etc.
    # We use a list for the counter to allow modification inside helper function (pass by reference)
    name_mapping = {}
    global_counter = [1]

    file_count = 0

    # Recursive traversal
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # 2. Traverse only .wav files
            if file.lower().endswith('.wav'):

                # Full path of source file
                src_file_path = os.path.join(root, file)

                # 4. Generate new filename
                # Get relative path components (e.g., ['aaa', 'bbb', 'ccc.wav'])
                rel_path = os.path.relpath(src_file_path, input_dir)
                path_parts = rel_path.split(os.sep)

                new_parts = []

                for i, part in enumerate(path_parts):
                    # Check if it is the last part (filename) to preserve extension logic
                    if i == len(path_parts) - 1:
                        fname, ext = os.path.splitext(part)
                        # Process filename part
                        new_fname = get_mapped_name(fname, name_mapping, global_counter)
                        new_parts.append(new_fname + ext)
                    else:
                        # Process folder parts
                        new_parts.append(get_mapped_name(part, name_mapping, global_counter))

                # Join with hyphen as requested
                new_filename = "-".join(new_parts)
                dest_file_path = os.path.join(dest_path, new_filename)

                # 3. Copy file
                try:
                    shutil.copy2(src_file_path, dest_file_path)
                    print(f"[OK] Copied: {src_file_path} -> {new_filename}")
                    file_count += 1
                except Exception as e:
                    print(f"[Err] Failed to copy {src_file_path}: {e}")

    print("-" * 30)
    print(f"Done. Total files copied: {file_count}")
    print(f"Output location: {dest_path}")

    # 5. Exit
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()

"@
$scriptPath = "$env:TEMP\temp_script.py"
Set-Content -Path $scriptPath -Value $pythonScript
python $scriptPath
Remove-Item $scriptPath  # 清理临时文件