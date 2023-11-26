import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    elif dataset == 'sunrgbd':
        n_classes = 38
        label_colours = get_sunrgbd_labels()
    elif dataset == 'scenenet':
        n_classes = 13
        label_colours = get_nyu13_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def sample_distribution(dataset, n=1000):
    n = min(n, len(dataset))
    m = 100

    mode = dataset.cfg.DATASET.MODE
    if mode == "RGBD":
        channels = 4
    elif mode == "RGB":
        channels = 3
    elif mode == "RGB_HHA":
        channels = 6
    else:
        raise ValueError('Dataset mode not implemented: {}'.format(dataset.mode))

    samples = np.zeros((m*n,channels))
    for it, i in tqdm(enumerate(np.random.choice(len(dataset), n))):
        sample = dataset[i]
        if isinstance(sample['image'], list):
            img = [np.asarray(img) for img in sample['image']]
            img = np.concatenate(img, axis=2)
        else:
            img = np.asarray(sample['image'])

        #Flatten image
        img = np.transpose(img, (1,2,0))
        img = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
        if img.shape[0]>m:
            pixel_i = np.random.choice(img.shape[0], m)
            samples[it*m:(it+1)*m,:] = img[pixel_i, :]

    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    m_max = np.max(samples, axis=0)
    median = np.median(samples, axis=0)

    # if dataset.mode == "RGBD":
    #     import matplotlib.pyplot as plt
    #     plt.hist(samples[:, -1], bins='auto')
    #     plt.title("Depth histogram")
    #     plt.show()

    return {'mean': mean, 'std': std, 'max': m_max, 'median':median, 'samples': samples}


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_sunrgbd_labels():
    return np.array([[31,119,180], [174,199,232], [255,127,14], [255,187,120], [44,160,44], [152,223,138], [214,39,40],
                     [255,152,150], [148,103,189], [197,176,213], [140,86,75], [196,156,148], [227,119,194], [247,182,210],
                     [127,127,127], [199,199,199], [188,189,34], [219,219,141], [23,190,207], [158,218,229], [141,211,199],
                     [255,255,179], [190,186,218], [251,128,114], [128,177,211], [253,180,98], [179,222,105], [252,205,229],
                     [217,217,217], [188,128,189], [204,235,197], [255,237,111], [228,26,28], [55,126,184], [77,175,74],
                     [152,78,163], [255,127,0], [255,255,51], [166,86,40], [247,129,191], [153,153,153], [98,30,21], [229,144,118],
                     [18,141,205], [8,60,82], [100,197,242], [97,175,175], [15,115,105], [156,157,161], [54,94,150], [152,51,52],
                     [119,151,61], [93,67,124], [54,134,159], [209,112,47], [129,151,197], [196,127,128], [172,196,132], [152,135,176],
                     [45,88,138], [88,149,76], [233,160,68], [193,47,50], [114,62,119], [125,128,127], [156,158,222], [115,117,181],
                     [74,85,132], [206,219,156], [181,207,107], [140,162,82], [99,121,57], [231,203,148], [231,186,82], [189,158,57], [140,109,49],
                     [231,150,156], [214,97,107], [173,73,74], [132,60,57], [222,158,214], [206,109,189], [165,81,148], [123,65,115], [0,0,0], [0,0,255]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

def get_nyu13_labels():
    return np.array([[  0,   0,   0],
       [  0,   0, 255], #BED
       [232,  88,  47], #BOOKS
       [  0, 217,   0], #CEILING
       [148,   0, 240], #CHAIR
       [222, 241,  23], #FLOOR
       [255, 205, 205], #FURNITURE
       [  0, 223, 228], #OBJECTS
       [106, 135, 204], #PAINTING
       [116,  28,  41], #SOFA
       [240,  35, 235], #TABLE
       [  0, 166, 156], #TV
       [249, 139,   0], #WALL
       [225, 228, 194]])  #WINDOWS