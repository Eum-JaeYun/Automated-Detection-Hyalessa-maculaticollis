#%%
"""
Code to convert files in various formats 
such as mp4, mov, 3gp, m4a, mp3 to 44.1 kHz, Mono channel, 16bit
"""
# Import library
import os
import glob
import ffmpy
import pydub
import shutil
import subprocess
from pydub import AudioSegment
#%%
def file_convert(inputdir):
    print(f"Processing Directory: {inputdir}")
    
    file_list = os.listdir(inputdir)

    # Statistical variables
    success_count = 0
    fail_count = 0
    skip_count = 0
    failed_files = []

    for filename in file_list:
        name, extension = os.path.splitext(filename)
        ext = extension.lower()
        
        input_file_path = os.path.join(inputdir, filename)
        output_file_path = os.path.join(inputdir, f"{name}.wav")

        # Skip if already a wav file
        if ext == '.wav':
            continue

        try:
            # Video formats (Using FFmpeg)
            if ext in ['.mp4', '.mov', '.3gp']:
                print(f"▶ Converting [FFmpeg]: {filename}")
                
                command = [
                    'ffmpeg', '-i', input_file_path, 
                    '-vn', '-ar', '44100', '-ac', '1', '-c:a', 'pcm_s16le', 
                    '-y', output_file_path,
                    '-loglevel', 'error'
                ]
                
                subprocess.run(command, check=True, shell=True) 
                success_count += 1
            
            # Audio Format (Using Pydub)
            elif ext in ['.m4a', '.mp3']:
                print(f"▶ Converting [Pydub]: {filename}")
                
                format_type = ext[1:]
                sound = AudioSegment.from_file(input_file_path, format=format_type)
                sound = sound.set_frame_rate(44100)
                sound = sound.set_channels(1)
                sound = sound.set_sample_width(2)
                sound.export(output_file_path, format='wav')
                success_count += 1
                
            else:
            # If it is not a wav file or a target for conversion
                skip_count += 1

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg Error on {filename}")
            fail_count += 1
            failed_files.append(filename)
            
        except Exception as e:
            print(f"General Error on {filename}: {e}")
            fail_count += 1
            failed_files.append(filename)

    # Print final report
    print("\n" + "="*30)
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed:  {fail_count}")
    
    if failed_files:
        print("\n[List of Failed Files]")
        for f in failed_files:
            print(f" - {f}")
    print("="*30 + "\n")

def move_wav_files(target_dir):
    destination_folder = os.path.join(target_dir, "wav_file")
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    wav_files = glob.glob(os.path.join(target_dir, "*.wav"))
    
    moved_count = 0
    for file_path in wav_files:
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(destination_folder, file_name)

        if os.path.abspath(file_path) == os.path.abspath(destination_path):
            continue

        try:
            shutil.move(file_path, destination_path)
            moved_count += 1
        except Exception as e:
            print(f"Error moving {file_name}: {e}")

    print(f"Moved {moved_count} wav files to '{destination_folder}' folder.")
#%%
if __name__ == '__main__':

    raw_data_path = input(r"Enter the path to the folder you want to work on here: ")

    print("-- File format Coverting Start --")
    file_convert(raw_data_path)
    move_wav_files(raw_data_path)
    print("-- File format Coverting Done --")