import os
import shutil

def copy_audio_files(source_dir, dest_dir, start_num=1, end_num=10):
    """
    Copy MP3 files in the specified number range from source directory to destination directory.

    Args:
        source_dir: Source directory path
        dest_dir: Destination directory path
        start_num: Starting file number (default 1)
        end_num: Ending file number (default 10)
    """

    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Copy files in the specified range
    for num in range(start_num, end_num + 1):
        file_name = f"jailbreakbench_prompt_{num}.mp3"
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)

        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path)  # Use copy2 to preserve metadata
                print(f"✅ Copied: {file_name} to {dest_dir}")
            except Exception as e:
                print(f"❌ Error copying {file_name}: {str(e)}")
        else:
            print(f"⚠️ File not found: {source_path}")

    print(f"🎉 Successfully copied jailbreakbench_prompt_{start_num} to jailbreakbench_prompt_{end_num} to {dest_dir}")

if __name__ == "__main__":
    source_directory = "audio/jailbreakbench"
    destination_directory = "experiment/origin_mp3"
    copy_audio_files(source_directory, destination_directory)
