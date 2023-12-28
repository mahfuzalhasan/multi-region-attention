import requests
import zipfile
import os
from tqdm import tqdm

# Define the URL of the ImageNet dataset
url = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/6799/4225553/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1704056930&Signature=NgLuC%2F5dx9Grj2c5c1z%2F0kswK6H4ZlxbX%2FqgOrRYP8dSAjsaiEsoUbs%2Ff3CSgqKVgP0x4r24viK5PMu2p3Htodb22rP2Mty3po9f%2FrLGHNINwOZEmUwAyaRJlvXBOzQxiG5MQloXknnZK7vGPUglhyPXHs9SAvA5lKlbr%2B%2BjMvMrkSkB9Ds%2FbOg5BOv%2FNZDKvHx%2Bfbc1AGjnW2CeaPHsKXVagX%2FEWGYWnXyKy0%2BIX8v4FUuvqTLmR8jNmb3tNLbc%2BZ%2B4sI2tvRygeI4q64hfvb8NBXBgQVeFxnVerNcgY7KBQEFP1NJQK%2FSqs4IjRKjRjizhZ8J3nF6h%2FCTa4lpFXg%3D%3D&response-content-disposition=attachment%3B+filename%3Dimagenet-object-localization-challenge.zip"

# Define the path where the dataset will be saved
path = "/home/abjawad/Documents"
filename = os.path.join(path, "imagenet-object-localization-challenge.zip")

# Download the dataset
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))
block_size = 1024
progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
with open(filename, 'wb') as file:
    for chunk in response.iter_content(chunk_size=block_size):
        if chunk:
            file.write(chunk)
            progress_bar.update(len(chunk))
progress_bar.close()

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
