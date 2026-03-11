#%%
"""
Split training and test data sets for K-Fold learning (8:2)
"""
import os
import shutil
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# %%
def split_and_copy(class_name):

    class_src_path = os.path.join(src_root, class_name)
    
    if not os.path.exists(class_src_path):
        print(f"Error: {class_src_path} folder does not exist.")
        return

    files = [f for f in os.listdir(class_src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files = np.array(files)
    
    print(f"[{class_name}] Total number of files: {len(files)}")

    # --- Divide by (8:2) ---
    # Train (80%) / Test (20%)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42, shuffle=True)
    
    datasets = {
        'Train': train_files, 
        'Test': test_files  
    }

    for split_type, file_list in datasets.items():
        # Create a path to save to (e.g. Input/Split_Data/Train/1_True)
        save_dir = os.path.join(dst_root, split_type, class_name)
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"  -> {split_type} Copying... ({len(file_list)})")
        
        for file_name in tqdm(file_list, desc=f"{class_name}-{split_type}", leave=False):
            src_file = os.path.join(class_src_path, file_name)
            dst_file = os.path.join(save_dir, file_name)
            shutil.copy(src_file, dst_file)
#%%
if __name__ == "__main__":
    
    src_root = input(r"Enter the path to the CNN model input file folder on here: ")
    dst_root = input(r"Enter the path to save the segmented data: ")
    classes = ['1_True', '0_False']  # Class Folder Name

    print("Start data splitting (8:2)...")

    for cls in classes:
        split_and_copy(cls)

    print("\nAll work is done!")
    print(f"Save location: {dst_root}")