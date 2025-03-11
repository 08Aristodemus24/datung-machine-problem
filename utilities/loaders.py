import requests
import zipfile
import tarfile
import os
import librosa
import numpy as np
import re

from concurrent.futures import ThreadPoolExecutor



def download_dataset(urls: list | set, data_dir="data"):
    

    # if directory already exists do nothing
    os.makedirs(f"./{data_dir}", exist_ok=True)

    def helper(url):
        file_name = url.split('/')[-1]

        print(file_name)
        response = requests.get(url, stream=True)

        # download the file given the urls
        with open(f"./{data_dir}/{file_name}", mode="wb") as file:
            for chunk in response.iter_content(chunk_size=10 * 1024):
                file.write(chunk)

    # concurrently download the files given url
    with ThreadPoolExecutor() as exe:
        exe.map(helper, urls)



def extract_all_files(tar_files: list, data_dir="data"):
    def helper(tar_file):
        print(f"extracting {tar_file}...")

        # extract tar file
        with tarfile.open(f'./{data_dir}/{tar_file}') as tar_ref:
            tar_ref.extractall('./data')

    # concurrently download the files given url
    with ThreadPoolExecutor() as exe:
        exe.map(helper, tar_files)



def load_labels(folders, DIR):
    def helper(folder):
        try:
            file_path = os.path.join(DIR, folder, "etc", "README")
            with open(file_path, "r") as file:
                lines = [line for line in file.readlines() if "gender" in line.lower()]
                file.close()

            # print(lines)

            # extract only the gender of the subject in meta data
            # print(lines[0].lower())
            string = re.search(r"(male|female|weiblich|männlich|unknown)", lines[0].lower())
            # print(string)
            if string:
                gender = string[0]
                if (gender == "male" or gender == "männlich"):
                    return folder, "male"
                elif (gender == "female" or gender == "weiblich"):
                    return folder, "female"
                else:
                    return folder, "unknown"
            
        except IndexError:
            return folder, "unknown"
        
        except FileNotFoundError:
            return folder, "unknown"

    with ThreadPoolExecutor() as exe:
        subjects_labels = list(exe.map(helper, folders))
        
        
    return subjects_labels



def load_audio(DIR: str, folders: list):
    """
    loads audio signals from each .wav file of each subject
    """

    def helper(folder):
    # for folder in folders:
        try:
            wavs_dir = os.path.join(DIR, folder, "wav")
            path_to_wavs = os.listdir(wavs_dir)

        # this is if a .wav file is not used as a directory so 
        # try flac 
        except FileNotFoundError:
            wavs_dir = os.path.join(DIR, folder, "flac")
            path_to_wavs = os.listdir(wavs_dir)

        finally:
            # create storage for list of signals to all be 
            # concatenated later
            ys = []

            # create figure, and axis
            # fig, axes = plt.subplots(nrows=len(path_to_wavs), ncols=1, figsize=(12, 30))
            # fig = plt.figure(figsize=(17, 5))
            for index, wav in enumerate(path_to_wavs):

                wav_path = os.path.join(wavs_dir, wav)
                # print(wav_path)

                # each .wav file has a sampling frequency is 16000 hertz 
                y, sr = librosa.load(wav_path, sr=None)

                # top_db is set to 20 representing any signal below
                # 20 decibels will be considered silence
                y_trimmed, _ = librosa.effects.trim(y, top_db=20)

                # append y to ys 
                ys.append(y_trimmed)

            # concatenate all audio signals into one final signal as 
            # this is all anyway recorded in the voice of the same gender
            final = np.concatenate(ys, axis=0)
            print(f"shape of final signal: {final.shape}")
            # print(f"shape of signal: {y.shape}")
            # print(f"shape of trimmed signal: {y_trimmed.shape}")
            # print(f"sampling rate: {sr}")
            # librosa.display.waveshow(final, alpha=0.5)

            # plt.tight_layout()
            # plt.show()

            return folder, final
        
    # concurrently load .wav files and trim  each .wav files
    # audio signal and combine into one signal for each subject 
    with ThreadPoolExecutor() as exe:
        signals = list(exe.map(helper, folders))
        
    return signals