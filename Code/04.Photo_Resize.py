#%%
"""
Resize to fit CNN model input size
"""
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
#%%
def process_single_image(file_info):

    img_path, save_path = file_info
    
    try:
        if save_path.exists():
            return "Skipped"

        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = img.resize(TARGET_SIZE, Image.BICUBIC)
            img.save(save_path)
            
        return "Success"
    except Exception as e:
        return f"Error: {e}"

def main():
    src_path = Path(SOURCE_FOLDER)
    dst_path = Path(SAVE_FOLDER)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[{SOURCE_FOLDER}] Searching for images in folder...")

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(src_path.rglob(ext))
    
    total_files = len(files)
    print(f"A total of {total_files} images were found.")
    
    if total_files == 0:
        print("There is no image. Please check the path.")
        return

    tasks = []
    for f in files:
        save_name = f.stem + ".png"
        save_file_path = dst_path / save_name
        tasks.append((f, save_file_path))

    print(f"Conversion starting! (Using {os.cpu_count()} CPU cores)")
    
    success_count = 0
    error_count = 0
    skipped_count = 0

    # Parallel execution with ProcessPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=total_files, unit="png"))
        
        for res in results:
            if res == "Success":
                success_count += 1
            elif res == "Skipped":
                skipped_count += 1
            else:
                error_count += 1

    print("-" * 30)
    print(f"Complete!")
    print(f"Success: {success_count}")
    print(f"Skip (already exists): {skipped_count}")
    print(f"Fail: {error_count}")
    print(f"Save Path: {dst_path.absolute()}")
#%%
if __name__ == '__main__':

    SOURCE_FOLDER = input(r"Enter the path to the png file on here: ")

    SAVE_FOLDER = input(r"Enter the path to the save folder on here: ")

    TARGET_SIZE = (224, 224) # CNN Model input size
    
    main()