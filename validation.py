import numpy as np
import torch

from config_cityscapes import config

from utils.metric import hist_info, compute_score, cal_mean_iou

# next two are likely not needed
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def compute_metric(results):
    hist = np.zeros((config.num_classes, config.num_classes))
    correct = 0
    labeled = 0
    count = 0
    for d in results:
        hist += d['hist']
        correct += d['correct']
        labeled += d['labeled']
        count += 1
    iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
    print(f'iou per class:{iou} miou:{mean_IoU}')
    result_dict = dict(mean_iou=mean_IoU, freq_iou=freq_IoU, mean_pixel_acc=mean_pixel_acc)
    return result_dict

def val_cityscape(epoch, val_loader, model):
    model.eval()
    sum_loss = 0
    all_results = []

    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            imgs = sample['image']      #B, 3, 1024, 2048
            gts = sample['label']       #B, 1024, 2048
            
            imgs = imgs.to(f'cuda:{model.device_ids[0]}', non_blocking=True)
            gts = gts.to(f'cuda:{model.device_ids[0]}', non_blocking=True)

            imgs_1, imgs_2 = imgs[:, :, :, :1024], imgs[:, :, :, 1024:]
            gts_1, gts_2 = gts[:, :, :1024], gts[:, :, 1024:]
            loss_1, out_1 = model(imgs_1, gts_1)
            loss_2, out_2 = model(imgs_2, gts_2)

            out = torch.cat((out_1, out_2), dim = 3)
            # mean over multi-gpu result
            loss = torch.mean(loss_1) + torch.mean(loss_2)
            #miou using torchmetric library
            # m_iou = cal_mean_iou(out, gts)

            score = out[0]      #1, C, H, W --> C, H, W = 19, H, W
            score = torch.exp(score)    
            score = score.permute(1, 2, 0)  #H,W,C
            pred = score.argmax(2)  #H,W
            
            pred = pred.detach().cpu().numpy()
            gts = gts[0].detach().cpu().numpy() #1, H, W --> H, W
            confusionMatrix, labeled, correct = hist_info(config.num_classes, pred, gts)
            results_dict = {'hist': confusionMatrix, 'labeled': labeled, 'correct': correct}
            all_results.append(results_dict)
            
            sum_loss += loss
            del loss
            if idx % config.val_print_stats == 0:
                print(f'sample {idx}')

        val_loss = sum_loss/len(val_loader)
        result_dict = compute_metric(all_results)

        print(f"\n ----------- evaluating in epoch:{epoch} ----------- \n")
        print('result: ',result_dict)
        print(f"#----------- epoch:{epoch} mean_iou:{result_dict['mean_iou']} -----------#")
        
        return val_loss, result_dict['mean_iou']
