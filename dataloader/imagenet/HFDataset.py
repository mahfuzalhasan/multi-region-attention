
import io
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image



class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        idx = int(idx)
        item = self.hf_dataset[idx]
        # Check if the image data needs to be wrapped with io.BytesIO
        if isinstance(item['image'], bytes):
            image = Image.open(item['image']).convert('RGB')
        else:
            # If the image is already a PIL Image, use it directly
            image = item['image'].convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, item['label']

