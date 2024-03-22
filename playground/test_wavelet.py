from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import os, sys
import torch.distributed as dist

sys.path.append('/home/ma906813/project2023/multi-region-attention')
import configs.config_imagenet
from dataloader.imagenet.build import build_transform


# Mock dist.get_rank for non-distributed training contexts
# Check if distributed training is initialized; if not, mock it
def mock_get_rank():
    return 0
if not dist.is_initialized():
    dist.get_rank = mock_get_rank


def calculate_energy(coefficients):
    return torch.sum(coefficients ** 2)

def load_image(path):
    # Path to your image file
    image_path = path

    # Using the same transformation as training data
    transform = build_transform(True, configs.config_imagenet.config)

    # Load the image
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0) # (B,C,H,W)

    # Move the tensor to GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    return image_tensor



if __name__ == '__main__':
    img_path = '/home/ma906813/project2023/multi-region-attention/playground/n01440764_10026.JPEG'
    
    # load image + apply transform
    image = load_image(img_path)
    print(f'Input image tensor: {image.shape}')

    xfm3 = DWTForward(J=3, wave='db3', mode='symmetric').cuda()
    xfm2 = DWTForward(J=2, wave='db2', mode='symmetric').cuda()
    xfm1 = DWTForward(J=1, wave='db1', mode='symmetric').cuda()
    xfm_seg = DWTForward(J=1, wave='haar', mode='zero').cuda()
    # ifm = DWTInverse(wave='db3', mode='symmetric').cuda()    



    # Yl, Yh = xfm3(image) 
    # print(f'Yl3 shape: {Yl.shape}')
    # print(f'No of Yh3: {len(Yh)}')
    # print(f'energy Yl3: {calculate_energy(Yl)}')
    # Yl, Yh = xfm2(image) 
    # print(f'Yl2 shape: {Yl.shape}')
    # print(f'No of Yh2: {len(Yh)}')
    # print(f'energy Yl2: {calculate_energy(Yl)}')
    Yl, Yh = xfm1(image) 
    print(f'Yl1 shape: {Yl.shape}')
    print(f'No of Yh1: {len(Yh)}')
    print(f'energy Yh1: {calculate_energy(Yh[0])}')
    print(f'energy Yl1: {calculate_energy(Yl)}')
    Yl, Yh = xfm_seg(image) 
    print(f'Yl1 shape: {Yl.shape}')
    print(f'No of Yh1: {len(Yh)}')
    print(f'energy Yh1: {calculate_energy(Yh[0])}')
    print(f'energy Yl1: {calculate_energy(Yl)}')
    


    # Plotting ----------------------------------------------------------------
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    ## Input image
    # B,C,H,W -> H,W,C for plotting
    axs[0].imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    ## Low-frequency component
    axs[1].imshow(Yl.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[1].set_title('Low-frequency Component')
    axs[1].axis('off')

    # print(f'Yh[2] shape: {Yh[2].squeeze()[0].permute(1, 2, 0).cpu().numpy().shape}')
    # exit()
    ## High-frequency component (first level)
    # print(len(Yh[2]))
    # for i in range(Yh[2].shape[1]):
    #     axs[2+i].imshow(Yh[2].squeeze()[i].permute(1, 2, 0).cpu().numpy())
    #     axs[2+i].set_title('High-frequency Component' + str(i))
    #     axs[2+i].axis('off')

    ## Reconstructed image
    # image_recon = ifm((Yl, Yh))
    # print(f'Reconstructed image shape: {image_recon.shape}')
    # axs[3+i].imshow(image_recon.squeeze().permute(1, 2, 0).cpu().numpy())
    # axs[3+i].set_title('Reconstructed Image')
    # axs[3+i].axis('off')

    plt.show()
    plt.savefig('wavelet.png')


