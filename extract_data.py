from utilities.loaders import extract_all_files
import os
import tarfile


if __name__ == "__main__":
    data_dir = "./data"
    tar_files = os.listdir(data_dir)

    print(len(tar_files))

    for tar_file in tar_files:
        # extract files from .tar file
        with tarfile.open(f'./{data_dir}/{tar_file}') as tar_ref:
            tar_ref.extractall('./data')

        # delete .tar file after extraction
        path = os.path.join(data_dir, tar_file)
        os.remove(path)


