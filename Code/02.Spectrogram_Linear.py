#%%
"""
Convert wav file to Linear Spectrogram
"""
import matplotlib
matplotlib.use('Agg') # Using non-GUI backend Agg

import librosa
import librosa.display
import soundfile as sf 
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

#%%
def check_file_properties(audio_path):

    try:
        info = sf.info(audio_path)
        if info.subtype != 'PCM_16':
            return False, f"Not 16-bit (Detected: {info.subtype})"
        return True, "OK"
    except Exception as e:
        return False, f"File Check Error: {e}"

def GetLinearSpectrogram(audio_path):
    SR = 44100
    TARGET_DURATION_SEC = 60
    TARGET_SAMPLES = TARGET_DURATION_SEC * SR

    # 16-bit Check
    is_valid, msg = check_file_properties(audio_path)
    if not is_valid:
        raise ValueError(msg)

    # Load (44.1kHz, Mono)
    y, sr = librosa.load(audio_path, sr=SR, mono=True, offset=0.0) 
    
    # Calculate actual audio length
    real_duration = librosa.get_duration(y=y, sr=sr)
    if real_duration > TARGET_DURATION_SEC :
        real_duration = TARGET_DURATION_SEC
    
    # Zero-padding or Truncate
    y_fixed = librosa.util.fix_length(y, size=TARGET_SAMPLES)
    
    # STFT (Hann Window, 75% Overlap)
    stft_result = librosa.stft(
        y_fixed, 
        n_fft=2048, 
        win_length=2048, 
        hop_length=512, 
        window='hann'
    )
    
    del y, y_fixed
    return sr, stft_result, real_duration

def save_spectrogram(sr, stft_result, name, duration, save_path, fig, ax, y_min, y_max):

    D = np.abs(stft_result)
    S_dB = librosa.amplitude_to_db(D, ref=np.max)
    
    ax.clear()
    ax.set_axis_off()
    
    # Draw Linear Spectrogram
    librosa.display.specshow(
        S_dB, 
        sr=sr, 
        n_fft=2048,
        win_length=2048, 
        hop_length=512, 
        y_axis='linear',
        x_axis='time', 
        ax=ax
    )
    
    # Frequency Range
    ax.set_ylim(y_min, y_max)
        
    save_name = f"{name}_dur_{duration:.2f}.png"
    full_path = os.path.join(save_path, save_name)
    
    fig.savefig(full_path, bbox_inches='tight', pad_inches=0)
    del D, S_dB

def make_spectogram(audio_path, save_path, y_min, y_max):
    wav_files = [f for f in os.listdir(audio_path) if f.endswith('.wav')]
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    fig.set_frameon(False)

    for i, f in enumerate(wav_files):
        path = os.path.join(audio_path, f)
        try:
            sr_data, stft_data, duration = GetLinearSpectrogram(path)
            
            save_spectrogram(sr_data, stft_data, f[:-4], duration, save_path, fig, ax, y_min, y_max)
            
            print(f'[{i+1}/{len(wav_files)}] {f} Done (Len: {duration:.2f}s)')
            del sr_data, stft_data
            
        except ValueError as ve:
            print(f'Skipped {f}: {ve}')
        except Exception as e:
            print(f'Error processing {f}: {e}')
        
        if i % 30 == 0: gc.collect()
    
    plt.close(fig)

#%%
if __name__ == '__main__':
    parent_folder = input(r"Enter the path to the mother folder containing the wav file on here: ")
    wav_file_path = input(r"Enter the path to the wav file on here: ")

    spectrogram_folder_name = "Image_Linear"
    
    spectrogram_save_path = os.path.join(parent_folder, spectrogram_folder_name)
    
    if not os.path.exists(spectrogram_save_path):
        print("Creating folder...")
        os.makedirs(spectrogram_save_path)
    
    print("Start making Linear spectrogram")
    
    make_spectogram(wav_file_path, spectrogram_save_path, 300, 20000) # Y-axis(Frequency axis) minimum and maximum settings
    
    print("Finish making spectrogram")