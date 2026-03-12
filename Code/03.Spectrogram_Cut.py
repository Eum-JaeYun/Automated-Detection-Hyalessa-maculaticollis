#%%
"""
Crop photos by unit time
"""
#  Import using module
import matplotlib
matplotlib.use('Agg')

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
#%%
def get_image_constants(image_dir_path):
    file_list = [f for f in os.listdir(image_dir_path) if f.endswith('.png')]
    
    if not file_list:
        raise FileNotFoundError("Png file doesn't exist in folder")
        
    # Open the first image to check the reference size
    with Image.open(os.path.join(image_dir_path, file_list[0])) as sample_image:
        width, height = sample_image.size
    
    # Calculating the X-axis (time): Calculating the pixel width to cut a 60-second long image into 5-second units
    crop_sec = 5 # If you want to change a crop_sec, Change this parameter
    full_duration = 60 # Total time of original image (fixed to 60 seconds)
    width_crop = int((width * crop_sec) / full_duration)
    
    return width, height, width_crop

def crop_images(image_dir_path, width, height, width_crop):
    zero_slice = []
    files = [f for f in os.listdir(image_dir_path) if f.endswith('.png')]
    
    FULL_DURATION = 60 

    for i, file_name in enumerate(files):
        full_path = os.path.join(image_dir_path, file_name)
        
        # Extract 'actual audio length' from filename (for skipping silent sections)
        try:
            base_name_part, dur_part = file_name.split('_dur_')
            real_duration = float(dur_part[:-4])
                
        except ValueError:
            real_duration = FULL_DURATION
            base_name_part = file_name[:-4]

        # Calculating the pixel position where valid audio ends
        valid_pixel_limit = int((width * real_duration) / FULL_DURATION)

        with Image.open(full_path) as img:
            
            save_folder_name = base_name_part 
            save_folder_path = os.path.join(image_dir_path, save_folder_name)
            
            if not os.path.exists(save_folder_path):
                os.mkdir(save_folder_path)
                
            coordinate = 0
            c = 0
            
            while coordinate < valid_pixel_limit:

                end_point = coordinate + width_crop

                if end_point > valid_pixel_limit :
                    break

                right = end_point
                
                box = (coordinate, 0, right, height)
                
                cropped_img = img.crop(box)
                
                save_name = f"{base_name_part}_{c}.png"
                cropped_img.save(os.path.join(save_folder_path, save_name))
                
                coordinate += width_crop
                c += 1
            
            print(f"[{i+1}/{len(files)}] Processed: {file_name} -> {c} slices (Limit: {real_duration}s)")
        
        # Avoid running out of memory
        if i % 20 == 0:
            gc.collect()
#%%
if __name__ == '__main__':

    target_dir = input(r"Enter the path to the png file on here: ")

    try:
        w, h, w_crop = get_image_constants(target_dir)
        print(f"Image Info - Width: {w}, Height: {h}, Crop Width (5s): {w_crop}")
        
        crop_images(target_dir, w, h, w_crop)
        print("All cropping finished successfully.")
        
    except Exception as e:
        print(f"Error occurred: {e}")