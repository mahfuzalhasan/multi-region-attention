import requests
import zipfile
import os
from tqdm import tqdm


# Define the path where the dataset will be saved
path = "/home/abjawad/Documents"
save_path = "/home/abjawad/Documents/imagenet"
filename = os.path.join(path, "imagenet-object-localization-challenge.zip")

# Extract the dataset
with zipfile.ZipFile(filename, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    total_files = len(file_list)
    extracted_files = 0
    progress_bar = tqdm(total=total_files, unit='file')
    for file in file_list:
        zip_ref.extract(file, path=path)
        extracted_files += 1
        progress_bar.update(1)
        progress_bar.set_description(f"Extracting: {file}")
    progress_bar.close()
