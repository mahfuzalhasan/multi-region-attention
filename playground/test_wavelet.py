from pytorch_wavelets import DWTForward, DWTInverse
import torch
import matplotlib.pyplot as plt
# from timm.data import create_transform


xfm = DWTForward(J=3, wave='db3', mode='symmetric').cuda()
ifm = DWTInverse(wave='db3', mode='symmetric').cuda()


from PIL import Image
from torchvision import transforms
import torch


def load_image(path):
    # Path to your image file
    image_path = path

    # Define the transformations: convert to grayscale, resize, convert to tensor
    # TODO: Use the same transformations as the training data
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((224,224)),  # Resize to 256x256 pixels
        transforms.ToTensor(),  # Convert to PyTorch Tensor
    ])

    # Load the image
    image = Image.open(image_path)

    # Apply the transformations
    image_tensor = transform(image)

    # Add a batch dimension (PyTorch expects batch, channel, height, width)
    image_tensor = image_tensor.unsqueeze(0)

    # Move the tensor to GPU if available
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    return image_tensor





if __name__ == '__main__':
    img_path = '/home/ma906813/project2023/multi-region-attention/playground/n01440764_10026.JPEG'
    image = load_image(img_path)
    # image = torch.randn(1, 1, 256, 256).cuda()
    print(f'Input image shape: {image.shape}')

    Yl, Yh = xfm(image) 
    print(f'Yl shape: {Yl.shape}')
    print(f'Yh shape: {len(Yh)}')
    print(f'Yh[0] shape: {Yh[0].shape}')
    print(f'Yh[1] shape: {Yh[1].shape}')
    print(f'Yh[2] shape: {Yh[2].shape}')

    image_recon = ifm((Yl, Yh))
    # print(f'Reconstructed image shape: {image_recon.shape}')


    # Assuming xfm and ifm are already defined and initialized
    # Assuming image, Yl, Yh, and image_recon are already computed

    # Move data to CPU and remove batch dimension for plotting
    image_cpu = image.squeeze().cpu()
    Yl_cpu = Yl.squeeze().cpu()
    Yh_cpu = Yh[0].squeeze().cpu()  # Taking the first set of high-frequency components for simplicity
    image_recon_cpu = image_recon.squeeze().cpu()

    # Plotting
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    # Input image
    axs[0].imshow(image_cpu, cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    # Low-frequency component
    axs[1].imshow(Yl_cpu, cmap='gray')
    axs[1].set_title('Low-frequency Component')
    axs[1].axis('off')

    # High-frequency component (first level)
    # Note: Adjust the visualization as needed, this is a simplification
    for i in range(len(Yh)):
        axs[2+i].imshow(Yh_cpu[i], cmap='gray')  # Assuming Yh[0] is LH (horizontal details) for demonstration
        axs[2+i].set_title('High-frequency Component' + str(i))
        axs[2+i].axis('off')

    # Reconstructed image
    # axs[3+i].imshow(image_recon_cpu, cmap='gray')
    # axs[3+i].set_title('Reconstructed Image')
    # axs[3+i].axis('off')

    plt.show()
    plt.savefig('wavelet.png')


