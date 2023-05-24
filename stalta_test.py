import numpy as np
import os
import argparse
import pandas as pd
import math
import logging
import pickle
import json
import bisect
import requests
from tqdm import tqdm

from calc import calc_intensity
from snr import snr_p
from utils import *

import torch
from torch.utils.data import DataLoader

import seisbench.data as sbd
import seisbench.generate as sbg

from obspy.signal.trigger import classic_sta_lta
from obspy.signal.trigger import trigger_onset

# import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--save_path', type=str, default='STALTA')
    parser.add_argument('--sample_tolerant', type=int, default=50)
    parser.add_argument("--threshold_short_window", type=int, default=20)
    parser.add_argument('--threshold_long_window', type=int, default=200)
    parser.add_argument('--threshold_lambda', type=float, default=4)
    parser.add_argument('--allTest', type=bool, default=False)
    parser.add_argument('--p_timestep', type=int, default=750)

    # dataset hyperparameters
    parser.add_argument('--dataset_opt', type=str, default='taiwan')
    parser.add_argument('--loading_method', type=str, default='full')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--snr_threshold', type=float, default=-1)
    parser.add_argument('--level', type=int, default=-1)
    parser.add_argument('--s_wave', type=bool, default=False)
    parser.add_argument('--instrument', type=str, default='all')
    parser.add_argument('--EEW', type=bool, default=False)
    parser.add_argument('--noise_sample', type=int, default=-1)

    opt = parser.parse_args()

    return opt

def toLine(save_path, precision, recall, fscore, mean, variance):
    token = "Eh3tinCwQ87qfqD9Dboy1mpd9uMavhGV9u5ohACgmCF"

    message = save_path + ' -> precision: ' + str(precision) + ', recall: ' + str(recall) + ', fscore: ' + str(fscore) + ', mean: ' + str(mean) + ', variance: ' + str(variance)
    
    try:
        url = "https://notify-api.line.me/api/notify"
        headers = {
            'Authorization': f'Bearer {token}'
        }
        payload = {
            'message': message
        }
        response = requests.request(
            "POST",
            url,
            headers=headers,
            data=payload
        )
        if response.status_code == 200:
            print(f"Success -> {response.text}")
    except Exception as e:
        print(e)

def set_generators(opt, ptime=None):
    cwbsn, tsmip, stead, cwbsn_noise, instance = load_dataset(opt)

    # split datasets
    if opt.dataset_opt == 'all':
        cwbsn_dev, cwbsn_test = cwbsn.dev(), cwbsn.test()
        tsmip_dev, tsmip_test = tsmip.dev(), tsmip.test()
        stead_dev, stead_test = stead.dev(), stead.test()
        cwbsn_noise_dev, cwbsn_noise_test = cwbsn_noise.dev(), cwbsn_noise.test()

        dev = cwbsn_dev + tsmip_dev + stead_dev + cwbsn_noise_dev
        test = cwbsn_test + tsmip_test + stead_test + cwbsn_noise_test
    elif opt.dataset_opt == 'cwbsn':
        cwbsn_dev, cwbsn_test = cwbsn.dev(), cwbsn.test()
        stead_dev, stead_test = stead.dev(), stead.test()

        dev = cwbsn_dev + stead_dev
        test = cwbsn_test + stead_test
    elif opt.dataset_opt == 'tsmip':
        tsmip_dev, tsmip_test = tsmip.dev(), tsmip.test()
        # stead_dev, stead_test = stead.dev(), stead.test()

        dev = tsmip_dev
        test = tsmip_test
        # dev = tsmip_dev + stead_dev
        # test = tsmip_test + stead_test
    elif opt.dataset_opt == 'stead':
        _, dev, test = stead.train_dev_test()
    elif opt.dataset_opt == 'instance':
        _, dev, test = instance.train_dev_test()
    elif opt.dataset_opt == 'redpan' or opt.dataset_opt == 'taiwan':
        cwbsn_dev, cwbsn_test = cwbsn.dev(), cwbsn.test()
        tsmip_dev, tsmip_test = tsmip.dev(), tsmip.test()
        cwbsn_noise_dev, cwbsn_noise_test = cwbsn_noise.dev(), cwbsn_noise.test()

        dev = cwbsn_dev + tsmip_dev + cwbsn_noise_dev
        test = cwbsn_test + tsmip_test + cwbsn_noise_test
    elif opt.dataset_opt == 'prev_taiwan':
        cwbsn_dev, cwbsn_test = cwbsn.dev(), cwbsn.test()
        tsmip_dev, tsmip_test = tsmip.dev(), tsmip.test()
        stead_dev, stead_test = stead.dev(), stead.test()

        dev = cwbsn_dev + tsmip_dev + stead_dev
        test = cwbsn_test + tsmip_test + stead_test
    elif opt.dataset_opt == 'EEW':        
        cwbsn_dev, cwbsn_test = cwbsn.dev(), cwbsn.test()
        tsmip_dev, tsmip_test = tsmip.dev(), tsmip.test()

        dev = cwbsn_dev + tsmip_dev
        test = cwbsn_test + tsmip_test

    print(f'total traces -> dev: {len(dev)}, test: {len(test)}')

    dev_generator = sbg.GenericGenerator(dev)
    test_generator = sbg.GenericGenerator(test)

    # set generator with or without augmentations
    phase_dict = ['trace_P_arrival_sample']
    if not opt.allTest:
        ptime = opt.p_timestep  
    augmentations = augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, keep_ori=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                ]
    
    dev_generator.add_augmentations(augmentations)
    test_generator.add_augmentations(augmentations)

    return dev_generator, test_generator

def calc_snr(data, isREDPAN=False):
    snr_batch = []
    gt = data['y'][:, 0] if not isREDPAN else data['X'][:, 3]

    for i in range(gt.shape[0]):
        # if not noise waveform, calculate log(SNR)
        tri = torch.where(gt[i] == 1)[0]
        if len(tri) != 0:
            snr_tmp = snr_p(data['ori_X'][i, 0, :].cpu().numpy(), gt[i].cpu().numpy())

            if snr_tmp is None:
                snr_batch.append(-1.0)
            else:
                tmp = np.log10(snr_tmp)
                if tmp < -1.0:
                    tmp = -1.0
                snr_batch.append(tmp)
                
        else:
            snr_batch.append(-9999)

    return snr_batch

def calc_inten(data, isREDPAN=False):
    intensity_batch = []
    gt = data['y'][:, 0] if not isREDPAN else data['X'][:, 3]

    for i in range(gt.shape[0]):
        tri = torch.where(gt[i] == 1)[0]
        if len(tri) == 0:
            intensity_batch.append(-1)
        else:
            intensity_tmp = calc_intensity(data['ori_X'][i, 0].numpy(), data['ori_X'][i, 1].numpy(), data['ori_X'][i, 2].numpy(), 'Acceleration', 100)

            intensity_batch.append(intensity_tmp)

    return intensity_batch

def convert_snr_to_level(snr_level, snr_total):
    res = []
    for i in snr_total:
        idx = bisect.bisect_right(snr_level, i)-1
        
        if idx < 0:
            idx = 0
            
        res.append(idx)

    return res

def convert_intensity_to_level(intensity_level, intensity_total):
    res = []
    for i in intensity_total:
        idx = intensity_level.index(i)

        res.append(idx)

    return res

def evaluation(pred, gt, snr_idx, snr_max_idx, intensity_idx, intensity_max_idx, sample_tolerant, isREDPAN_dataset=False):
    tp, fp, tn, fn = 0, 0, 0, 0 
    multiple_trigger = 0
    diff = []
    abs_diff = []
    res = []

    # snr stat
    snr_stat = {}
    for i in range(snr_max_idx):
        snr_stat[str(i)] = []
        
    # intensity stat
    intensity_stat = {}
    for i in range(intensity_max_idx):
        intensity_stat[str(i)] = []

    # stat case-by-case
    case_stat = {}
    case_stat['snr'], case_stat['intensity'], case_stat['res'] = [], [], []

    for i in range(len(pred)):
        pred_isTrigger = False
        gt_isTrigger = False
        gt_trigger = 0

        if isREDPAN_dataset and False:
            snr_cur = 0
            intensity_cur = 0
        else:
            snr_cur = snr_idx[i]
            intensity_cur = intensity_idx[i]

        if not np.all(gt[i] == 0):
            gt_isTrigger = True            
            gt_trigger = np.argmax(gt[i])
            if gt[i][gt_trigger] < 0.3:
                gt_isTrigger = False
                gt_trigger = 0

        if len(pred[i]) == 0:
            pred_isTrigger = False
            pred_trigger = 0
        else:
            pred_isTrigger = True
            if len(pred[i]) > 1:
                multiple_trigger += 1

            if not gt_isTrigger:
                pred_trigger = pred[i][0][0]
            else:
                nearest = 0
                tmp_diff = 3000
                for pp in pred[i]:
                    if abs(pp[0] - gt_trigger) < tmp_diff:
                        tmp_diff = abs(pp[0] - gt_trigger)
                        nearest = pp[0]
                
                pred_trigger = nearest

        left_edge = (gt_trigger - sample_tolerant) if (gt_trigger - sample_tolerant) >= 0 else 0
        right_edge = (gt_trigger + sample_tolerant) if (gt_trigger + sample_tolerant) <= 3000 else 2999

        # case positive 
        if (pred_trigger >= left_edge) and (pred_trigger <= right_edge) and (pred_isTrigger) and (gt_isTrigger):
            tp += 1
            res.append('tp')
            snr_stat[str(snr_cur)].append('tp')
            intensity_stat[str(intensity_cur)].append('tp')
        elif (pred_isTrigger):
            fp += 1
            res.append('fp')
            snr_stat[str(snr_cur)].append('fp')
            intensity_stat[str(intensity_cur)].append('fp')

        # case negative
        if (not pred_isTrigger) and (gt_isTrigger):
            fn += 1
            res.append('fn')
            snr_stat[str(snr_cur)].append('fn')
            intensity_stat[str(intensity_cur)].append('fn')
        elif (not pred_isTrigger) and (not gt_isTrigger):
            tn += 1
            res.append('tn')
            snr_stat[str(snr_cur)].append('tn')
            intensity_stat[str(intensity_cur)].append('tn')

        if gt_isTrigger and pred_isTrigger:
            diff.append(pred_trigger-gt_trigger)
            abs_diff.append(abs(pred_trigger-gt_trigger))

        case_stat['snr'].append(str(snr_cur))
        case_stat['intensity'].append(str(intensity_cur))
        case_stat['res'].append(res[i])

    return tp, fp, tn, fn, diff, abs_diff, res, snr_stat, intensity_stat, case_stat, multiple_trigger

def inference(opt, test_loader, short_window, long_window, threshold_lambda):
    # 先把整個 test set 的預測結果都跑完一遍

    pred = []
    gt = []
    snr_total = []
    intensity_total = []
    isREDPAN =  False

    with tqdm(test_loader) as epoch:
        for data in epoch:          
            try:
                if not opt.dataset_opt == 'REDPAN_dataset':
                    snr_total += calc_snr(data, isREDPAN)
                    intensity_total += calc_inten(data, isREDPAN)
                    
                with torch.no_grad():
                    if opt.dataset_opt == 'REDPAN_dataset':
                        wf, psn, mask = data

                        out = classic_sta_lta(wf[0, 0], short_window, long_window)
                        trigger = trigger_onset(out, threshold_lambda, 1)
                        pred += [trigger]
                        gt += [psn[i, 0] for i in range(wf.shape[0])]

                    else:
                        out = classic_sta_lta(data['X'][0, 0], short_window, long_window)
                        trigger = trigger_onset(out, threshold_lambda, 1)

                        pred += [trigger]
                        target = data['y'][:, 0].squeeze().numpy()
                        gt += [target]
            except:
                continue
            
    return pred, gt, snr_total, intensity_total

def score(pred, gt, snr_total, intensity_total, opt, isTest=False):
    # 依照 snr 不同分別計算數據，先將原本的 snr level 轉換成對應 index
    # snr_level = list(np.arange(0.0, 3.5, 0.25)) + list(np.arange(3.5, 5.5, 0.5))
    snr_level = [-9999] + list(np.arange(-1.0, 0.0, 0.5)) + list(np.arange(0.0, 3.5, 0.25)) + list(np.arange(3.5, 5.5, 0.5))
    intensity_level = [-1, 0, 1, 2, 3, 4, 5, 5.5, 6, 6.5, 7]

    if not opt.dataset_opt == 'REDPAN_dataset':    
        snr_idx = convert_snr_to_level(snr_level, snr_total)
        intensity_idx = convert_intensity_to_level(intensity_level, intensity_total)

    if not opt.dataset_opt == 'REDPAN_dataset':
        tp, fp, tn, fn, diff, abs_diff, res, snr_stat, intensity_stat, case_stat, multiple_trigger = evaluation(pred, gt, snr_idx, len(snr_level), intensity_idx, len(intensity_level), opt.sample_tolerant, isREDPAN_dataset=False)
    else:
        snr_idx, intensity_idx = 0, 0
        tp, fp, tn, fn, diff, abs_diff, res, snr_stat, intensity_stat, case_stat, multiple_trigger = evaluation(pred, gt, snr_idx, len(snr_level), intensity_idx, len(intensity_level), opt.sample_tolerant, isREDPAN_dataset=False)

    # print('tp=%d, fp=%d, tn=%d, fn=%d' %(tp, fp, tn, fn))

    # statisical  
    precision = tp / (tp+fp) if (tp+fp) != 0 else 0
    recall = tp / (tp+fn) if (tp+fn) != 0 else 0
    fpr = fp / (tn+fp) if (tn+fp) != 0 else 100
    fscore = 2*precision*recall / (precision+recall) if (precision+recall) != 0 else 0
    # mcc = (tp*tn-fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    logging.info('======================================================')
    logging.info('threshold_short_window: %.2f' %(opt.threshold_short_window))
    logging.info('threshold_long_window: %d' %(opt.threshold_long_window))
    logging.info('threshold_lambda: %d' %(opt.threshold_lambda))
    logging.info('TPR=%.4f, FPR=%.4f, Precision=%.4f, Fscore=%.4f' %(recall, fpr, precision, fscore))
    logging.info('tp=%d, fp=%d, tn=%d, fn=%d' %(tp, fp, tn, fn))
    logging.info('abs_diff=%.4f, diff=%.4f' %(np.mean(abs_diff)/100, np.mean(diff)/100))
    logging.info('trigger_mean=%.4f, trigger_std=%.4f' %(np.mean(diff)/100, np.std(diff)/100))
    # logging.info('MCC=%.4f' %(mcc))
    logging.info('RMSE=%.4f, MAE=%.4f' %(np.sqrt(np.mean(np.array(diff)**2))/100, np.mean(abs_diff)/100))
    logging.info('Multiple trigger=%d' %(multiple_trigger))

    if isTest:
        toLine(opt.save_path, precision, recall, fscore, np.mean(diff)/100, np.std(diff)/100)

    return fscore, abs_diff, diff, snr_stat, intensity_stat, case_stat

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    opt = parse_args()

    output_dir = os.path.join('./results', opt.save_path)
    model_dir = output_dir
    if opt.level == -1:
        level = 'all'
    else:
        level = str(opt.level)

    if not opt.allTest:
        output_dir = os.path.join(output_dir, opt.dataset_opt)
    else:
        output_dir = os.path.join(output_dir, f"allTest_{opt.dataset_opt}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subpath = 'threshold'
    if opt.level != -1:
        subpath = subpath + '_' + str(opt.level)
    if opt.p_timestep != 750:
        subpath = subpath + '_' + str(opt.p_timestep)
    if opt.allTest:
        subpath = subpath + '_allCase_testing_' + str(opt.level)

    subpath = f"{subpath}_{opt.threshold_short_window}_{opt.threshold_long_window}"
    subpath = subpath + '.log'
    print('logpath: ', subpath)
    log_path = os.path.join(output_dir, subpath)

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        filename=log_path, 
                        filemode='a', 
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S",)
    print(opt.save_path)

    # load datasets
    if not opt.allTest:
        print('loading datasets')
        if not opt.dataset_opt == 'REDPAN_dataset':
            dev_generator, test_generator = set_generators(opt)
            dev_loader = DataLoader(dev_generator, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
            test_loader = DataLoader(test_generator, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        else:
            basedir = '/mnt/disk4/weiwei/seismic_datasets/REDPAN_30S_pt/'
            dev_set, test_set = REDPAN_dataset(basedir, 'val', 1.0, 'REDPAN'), REDPAN_dataset(basedir, 'test', 1.0, 'REDPAN')
            
            # create dataloaders
            print('creating dataloaders')
            dev_loader = DataLoader(dev_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
            test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    cnt = 0
    front = opt.sample_tolerant
    back = opt.sample_tolerant

    if not opt.allTest:
        pred, gt, snr_total, intensity_total = pred, gt, snr_total, intensity_total = inference(opt, test_loader, opt.threshold_short_window, opt.threshold_long_window, opt.threshold_lambda)
        logging.info('======================================================')

        fscore, abs_diff, diff, snr_stat, intensity_stat, case_stat = score(pred, gt, snr_total, intensity_total, opt, isTest=False)
        print('fscore: %.4f' %(fscore))

        with open(os.path.join(output_dir, 'abs_diff_'+str(opt.level)+'.pkl'), 'wb') as f:
            pickle.dump(abs_diff, f)

        with open(os.path.join(output_dir, 'diff_'+str(opt.level)+'.pkl'), 'wb') as f:
            pickle.dump(diff, f)

        with open(os.path.join(output_dir, 'snr_stat_'+str(opt.level)+'.json'), 'w') as f:
            json.dump(snr_stat, f)

        with open(os.path.join(output_dir, 'intensity_stat_'+str(opt.level)+'.json'), 'w') as f:
            json.dump(intensity_stat, f)

        with open(os.path.join(output_dir, 'case_stat_'+str(opt.level)+'.json'), 'w') as f:
            json.dump(case_stat, f)

    # 將 p arrival 固定在多個不同時間點，分別得到實驗結果
    if opt.allTest:
        logging.info('configs: ')
        logging.info(opt)
        logging.info('dataset: %s' %(opt.dataset_opt))
        
        print('Start testing...')
        ptime_list = [750, 1500, 2000, 2500, 2750]
      
        for ptime in ptime_list:
            print('='*50)
            print(f"ptime: {ptime}")
            new_output_dir = os.path.join(output_dir, str(ptime))
            if not os.path.exists(new_output_dir):
                os.makedirs(new_output_dir)

            _, test_generator = set_generators(opt, ptime=ptime)
            test_loader = DataLoader(test_generator, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

            # start predicting on test set
            logging.info('======================================================')
            logging.info('Inference on testing set, ptime: %d' %(ptime))
            pred, gt, snr_total, intensity_total = pred, gt, snr_total, intensity_total = inference(opt, test_loader, opt.threshold_short_window, opt.threshold_long_window, opt.threshold_lambda)
            fscore, abs_diff, diff, snr_stat, intensity_stat, case_stat = score(pred, gt, snr_total, intensity_total, opt, isTest=False)
            print(f"ptime: {ptime}, fscore: {fscore}")
            logging.info('======================================================')

            with open(os.path.join(new_output_dir, 'test_abs_diff_'+str(opt.level)+'.pkl'), 'wb') as f:
                pickle.dump(abs_diff, f)

            with open(os.path.join(new_output_dir, 'test_diff_'+str(opt.level)+'.pkl'), 'wb') as f:
                pickle.dump(diff, f)

            with open(os.path.join(new_output_dir, 'test_snr_stat_'+str(opt.level)+'.json'), 'w') as f:
                json.dump(snr_stat, f)

            with open(os.path.join(new_output_dir, 'test_intensity_stat_'+str(opt.level)+'.json'), 'w') as f:
                json.dump(intensity_stat, f)

            with open(os.path.join(new_output_dir, 'test_case_stat_'+str(opt.level)+'.json'), 'w') as f:
                json.dump(case_stat, f)
