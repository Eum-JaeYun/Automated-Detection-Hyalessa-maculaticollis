#%%
"""
Data Labeling (True/False Divide)
"""
import os
import numpy as np
import shutil
import pandas as pd
from natsort import natsorted
#%%
def join_path(mother, child, photo = None):
    if photo is None:
        return os.path.join(mother, child)
    else:
        return os.path.join(mother, child, photo)

def find_photo_name(sub_path):
    photo_name_list = os.listdir(sub_path)
    return natsorted(photo_name_list)

def moving(code, path, True_path = True_path, False_path = False_path):
    TF = annotation_file[annotation_file['PngCode'] == code]['T/F'].values[0]
    print(TF)
    if  TF == 1 :
        shutil.move(path,True_path)
        file_identity = 'True'
    else :
        shutil.move(path,False_path)
        file_identity = 'False'
    print(f"Moved {code} to {file_identity}")
#%%
if __name__ == "__main__":

    annotation_file_path = input(r"Enter the path to the annotation_file on here: ")
    annotation_file = pd.read_excel(annotation_file_path,sheet_name='Nonoverlap', usecols=['PngCode', 'T/F'])

    mother_directory_path = input(r"Enter the path to the image folder on here: ")
    True_path = input(r"Enter the path to the true file save folder on here: ")
    False_path = input(r"Enter the path to the false file save folder on here: ")
    
    sub_directory_list = os.listdir(mother_directory_path)

    resnet_input_png_path_dir = {}

    photo_name_list = find_photo_name(mother_directory_path)

    for photo in photo_name_list:
        code = str(photo.split('.')[0])
        resnet_input_png_path_dir[code] = join_path(mother_directory_path, photo)

    for key, value in resnet_input_png_path_dir.items():
        moving(str(key),value)