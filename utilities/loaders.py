import requests
import zipfile
import tarfile
import os
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