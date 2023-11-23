import os
import cv2
import argparse
import numpy as np

import argparse
from datetime import datetime
import time
from tqdm import tqdm
from timm.models.layers import to_2tuple

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import multiprocessing as mp

# from configs.config_cityscapes import config
from utils.pyt_utils import ensure_dir
from utils.visualize import print_iou, show_img
# from engine.evaluator import Evaluator      # *REMOVE*
from utils.logger import get_logger        # *REPLACE with new logger*
from utils.metric import hist_info, compute_score
from utils.pyt_utils import load_model, ensure_dir
from utils.transforms import pad_image_to_shape

from models.builder import EncoderDecoder as segmodel

# from dataloader.cfg_defaults import get_cfg_defaults
from dataloader.cityscapes_dataloader import CityscapesDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = get_logger()

class SegEvaluator(object):
    def __init__(self, dataset, class_num, norm_mean, norm_std, network, multi_scales, 
                is_flip, devices, verbose=False, save_path=None, show_image=False):
        self.eval_time = 0
        self.dataset = dataset
        self.ndata = self.dataset.get_length() # total test file length
        self.class_num = class_num
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.multi_scales = multi_scales
        self.is_flip = is_flip
        self.network = network
        self.devices = devices

        self.context = mp.get_context('spawn')
        self.val_func = None
        self.results_queue = self.context.Queue(self.ndata)

        self.verbose = verbose
        self.save_path = save_path
        if save_path is not None:
            ensure_dir(save_path)
        self.show_image = show_image
    
    def run(self, model_path, model, log_file):
        results = open(log_file, 'a')
        # logger.info("Load Model: %s" % model)
        self.val_func = load_model(self.network, os.path.join(model_path, model))
        if len(self.devices) == 1:
            result_line = self.single_process_evaluation()
        else:
            result_line, _ = self.multi_process_evaluation()

        results.write('Model: ' + model + '\n')
        results.write(result_line)
        results.write('\n')
        results.flush()

        results.close()
    
    def single_process_evaluation(self):
        start_eval_time = time.perf_counter()

        # logger.info('GPU %s handle %d data.' % (self.devices[0], self.ndata))
        all_results = []
        
        for idx in tqdm(range(self.ndata)):
            dd = self.dataset[idx]      # one data/image
            results_dict = self.func_per_iteration(dd, self.devices[0])
            all_results.append(results_dict)
        print("all results single process: ", all_results)
        result_line = self.compute_metric(all_results)
        # logger.info('Evaluation Elapsed Time: %.2fs' % (time.perf_counter() - start_eval_time))
        return result_line
    
    def multi_process_evaluation(self):
        # TODO: check this function'''
        start_eval_time = time.perf_counter()
        nr_devices = len(self.devices)
        stride = int(np.ceil(self.ndata / nr_devices))
        
        ''' data is equally distributed to available devices'''
        
        procs = []
        for d in range(nr_devices):
            e_record = min((d + 1) * stride, self.ndata)    #500
            shred_list = list(range(d * stride, e_record))
            device = self.devices[d]
            # logger.info('GPU %s handle %d data.' % (device, len(shred_list)))

            p = self.context.Process(target=self.worker,
                                     args=(shred_list, device)) # ([0-500], 0), ([500-1000], 1)    
            procs.append(p)

        for p in procs:
            p.start()

        all_results = []
        for _ in tqdm(range(self.ndata)):
            t = self.results_queue.get()
            all_results.append(t)
            if self.verbose:
                self.compute_metric(all_results)

        for p in procs:
            p.join()

        result_line, output_dict = self.compute_metric(all_results)
        # logger.info('Evaluation Elapsed Time: %.2fs' % (time.perf_counter() - start_eval_time))
        return result_line, output_dict
    
    # K threads will run this function in parallel using the passed arguments
    def worker(self, shred_list, device):
        ''' multi-process worker'''
        start_load_time = time.time()
        # logger.info('Load Model on Device %d: %.2fs' % (device, time.time() - start_load_time))

        # change for different dataset
        for idx in shred_list:
            dd = self.dataset[idx]
            results_dict = self.func_per_iteration(dd, device)
            self.results_queue.put(results_dict)
    
    def func_per_iteration(self, data, device):
        ''' evaluates on a single image and returns result dict'''
        img = data['image']      # H, W, C
        label = data['label']  
        name = data['id']

        img = torch.permute(img, (1, 2, 0))
        img = img.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        pred = self.sliding_eval(img, config.EVAL.eval_crop_size, config.EVAL.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.DATASET.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path+'_color', fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def sliding_eval(self, img, crop_size, stride_rate, device=None):
        ''' sliding evaluation for a single image of any size '''
        crop_size = to_2tuple(crop_size)
        ori_rows, ori_cols, _ = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:
            # img resize to scale
            img_scale = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            processed_pred += self.scale_process(img_scale, (ori_rows, ori_cols),
                                                        crop_size, stride_rate, device)

        # for all pixels, select the class with maximum score
        pred = processed_pred.argmax(2) # 480, 640, 1
        return pred
    
    def scale_process(self, img, ori_shape, crop_size, stride_rate, device=None):
        ''' process a single image of a certain scale '''
        new_rows, new_cols, c = img.shape
        if new_cols <= crop_size[1] and new_rows <= crop_size[0]:
            input_data, margin = self.process_image(img, crop_size)
            score = self.val_func_process_rgbX(input_data, device) 
            # if scale < 1, then discard score for padded portions.
            score = score[:, margin[0]:(score.shape[1] - margin[1]), margin[2]:(score.shape[2] - margin[3])]
        
        else: 
            stride = (int(np.ceil(crop_size[0] * stride_rate)), int(np.ceil(crop_size[1] * stride_rate)))
            img_pad, margin = pad_image_to_shape(img, crop_size, cv2.BORDER_CONSTANT, value=0)
            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride[0])) + 1
            c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride[1])) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(device)


            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride[0]
                    s_y = grid_yidx * stride[1]
                    e_x = min(s_x + crop_size[0], pad_cols)
                    e_y = min(s_y + crop_size[1], pad_rows)
                    s_x = e_x - crop_size[0]
                    s_y = e_y - crop_size[1]
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    input_data, tmargin = self.process_image(img_sub, crop_size)

                    temp_score = self.val_func_process(input_data, device)
                    
                    temp_score = temp_score[:, tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                            tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)      # H, W, C
        data_output = cv2.resize(score.cpu().numpy(), (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_LINEAR)

        return data_output


    def val_func_process(self, input_data, device=None):
        ''' generates output prediction for a single image '''        
        input_data = np.ascontiguousarray(input_data[None, :, :, :], dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device) # 1, C, H, W, 

        with torch.cuda.device(input_data.get_device()):
            self.val_func.to(input_data.get_device())
            self.val_func.eval()

            with torch.no_grad():
                score = self.val_func(input_data)    # 1, C, H, W
                score = score[0]                # C, H, W
                if self.is_flip:
                    input_data = input_data.flip(-1)
                    score_flip = self.val_func(input_data)
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                score = torch.exp(score)            # e^score
        
        return score


    # when scaled image size <= crop_size (original size) condition
    def process_image(self, img, crop_size=None):
        ''' process and prepare a single image for inference '''
        p_img = img
    
        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), amodal_xis=2)
    
    
        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)   # C, H, W
            # margin --> length of padding on four side
            return p_img, margin
    
        p_img = p_img.transpose(2, 0, 1) # 3 H W
    
        return p_img
    
    def compute_metric(self, results):
        ''' computes metrics over results'''
        hist = np.zeros((config.DATASET.num_classes, config.DATASET.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1
        
        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_dict = dict(mean_iou=mean_IoU, freq_iou=freq_IoU, mean_pixel_acc=mean_pixel_acc)
        print('result dict: ',result_dict)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)
        return result_line, result_dict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Segformer Training')
    parser.add_argument('config', help='dataset specific train config file path, more details can be found in configs/')

    args = parser.parse_args()
    config_filename = args.config.split('/')[-1].split('.')[0] 
    
    if config_filename == 'config_cityscapes':
        from configs.config_cityscapes import config
    else:
        raise NotImplementedError

    # print(f'config:{config}')
    # exit()
    logger = get_logger()

    os.environ['MASTER_PORT'] = '169710'
    
    run_id = datetime.today().strftime('%m-%d-%y_%H%M')
    print(f'$$$$$$$$$$$$$ run_id:{run_id} $$$$$$$$$$$$$')
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.SYSTEM.seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    dataset = CityscapesDataset(config, split='val')
    
    # loading our config file here on --------------------
    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)

    all_devices = config.SYSTEM.device_ids

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.DATASET.num_classes, 
                                config.DATASET.norm_mean,
                                config.DATASET.norm_std, network,
                                config.EVAL.eval_scale_array, 
                                config.EVAL.eval_flip,
                                 all_devices, verbose=False, save_path=None,
                                 show_image = False)
        saved_model_path = config.WRITE.checkpoint_dir
        saved_model_names = ["model_435_11-06-23_0957.pth", "model_495_11-06-23_0957.pth"]
        # saved_model_names = ["model_230_attn_merge_mhms_11-01-23_1037.pth"]
        
        for i in range(len(saved_model_names)):
            name = saved_model_names[i][:saved_model_names[i].rindex('.')]+'.log'
            log_file = os.path.join(config.WRITE.log_dir, name)
            print('## ----------log file is ', log_file)
            print(f" ####### \n Testing with model {saved_model_names[i]} \n #######")
            segmentor.run(saved_model_path, saved_model_names[i], log_file)
