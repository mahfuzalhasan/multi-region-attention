import requests
from tqdm import tqdm
import os

def download_file(url, filename):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Start the download
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Check for request errors

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # Use a larger block size of 1MB

        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")

        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:  # Filter out keep-alive new chunks
                    file.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()

# Define the URL of the ImageNet dataset
url = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/6799/4225553/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1704061320&Signature=TEYsta0BOIIfbTZLR%2FQgLAP1IkSd6qUNs%2BEETbdx2yDGsZaBdFm4rdjhO5anvFUksnTuaGs2ar99vV2SG8SeHwxlD6GiSfNYzVay2FmKwJ36bqwQGrXPSlF0Cb3ncIeI0epFNtTInd9PrkMKC9t0LZZQ%2F9zleUU7qlpOuAUex6CyiL8JM%2Bt6d6PLmBUoNUK6Tp9OBgPvnZITXZarpSOLp6w6b6Zct3hidgzyfFt8mYGwEpicDUvbuDQYMS1qsrqL5w9KK%2BLEX7XwaN2T2ZBkd2%2BR5PAX3DNcCI7U8SYXsjFEmUMNQS6B%2BPvprFSKDlfLhJbBuLZB9uoqEcAjCqB22Q%3D%3D&response-content-disposition=attachment%3B+filename%3Dimagenet-object-localization-challenge.zip"

# Define the path where the dataset will be saved
path = "/project"
filename = os.path.join(path, "imagenet-object-localization-challenge.zip")

# Download the dataset
download_file(url, filename)
