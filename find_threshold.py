import numpy as np
import os
import argparse
import pandas as pd
import math
import logging
import pickle
import json
import time
import bisect
import requests
from tqdm import tqdm

import sys
sys.path.append('./RED-PAN')
from REDPAN_dataset import *

sys.path.append('./eqt')
from load_eqt import *

from calc import calc_intensity
from snr import snr_p
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import seisbench.data as sbd
import seisbench.generate as sbg

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_path", type=str, default='tmp')
    parser.add_argument('--load_last', type=bool, default=False)
    parser.add_argument('--load_specific_model', type=str, default='None')
    parser.add_argument("--threshold_type", type=str, default='all')
    parser.add_argument('--threshold_prob_start', type=float, default=0.15)
    parser.add_argument('--threshold_prob_end', type=float, default=0.9)
    parser.add_argument('--threshold_trigger_start', type=int, default=5)
    parser.add_argument('--threshold_trigger_end', type=int, default=45)
    parser.add_argument('--sample_tolerant', type=int, default=50)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--p_timestep', type=int, default=750)
    parser.add_argument('--allTest', type=bool, default=False)
    parser.add_argument('--s_test', type=bool, default=False)

    # dataset hyperparameters
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--snr_threshold', type=float, default=-1)
    parser.add_argument('--level', type=int, default=-1)
    parser.add_argument('--s_wave', type=bool, default=False)
    parser.add_argument('--instrument', type=str, default='all')
    parser.add_argument('--EEW', type=bool, default=False)
    parser.add_argument('--noise_sample', type=int, default=-1)
    
    parser.add_argument('--EEW_allTest', type=bool, default=False)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--batch_size", type=int, default=100)

    # REDPAN dataset
    parser.add_argument('--load_to_ram', type=bool, default=False)

    # seisbench options
    parser.add_argument('--model_opt', type=str, default='none')
    parser.add_argument('--loss_weight', type=float, default=50)
    parser.add_argument('--dataset_opt', type=str, default='taiwan')
    parser.add_argument('--loading_method', type=str, default='full')
    parser.add_argument('--normalize_opt', type=str, default='peak')
    parser.add_argument('--isConformer', type=bool, default=False)
    
    # custom hyperparameters
    parser.add_argument('--conformer_class', type=int, default=16)
    parser.add_argument('--d_ffn', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=12)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--encoder_type', type=str, default='conformer')
    parser.add_argument('--decoder_type', type=str, default='crossattn')
    parser.add_argument('--query_type', type=str, default='pos_emb')
    parser.add_argument('--intensity_MT', type=bool, default=False)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--label_type', type=str, default='p')
    
    # MGAN block
    parser.add_argument('--dim_spectrogram', type=str, default='1D')
    parser.add_argument('--MGAN_normtype', type=str, default='mean')
    parser.add_argument('--MGAN_l', type=int, default=10)

    # pretrained embedding
    parser.add_argument('--emb_type', type=str, default='transformer_MLM')
    parser.add_argument('--emb_model_opt', type=str, default='transformer')
    parser.add_argument('--emb_d_ffn', type=int, default=256)
    parser.add_argument('--emb_layers', type=int, default=4)
    parser.add_argument('--pretrained_emb', type=str)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_class', type=int, default=16)

    # GRADUATE model
    parser.add_argument('--cross_attn_type', type=int, default=2)
    parser.add_argument('--n_segmentation', type=int, default=2)
    parser.add_argument('--output_layer_type', type=str, default='fc')
    parser.add_argument('--rep_KV', type=str, default='False')
    parser.add_argument('--segmentation_ratio', type=float, default=0.35)
    parser.add_argument('--seg_proj_type', type=str, default='crossattn')
    parser.add_argument('--recover_type', type=str, default='crossattn')
    parser.add_argument('--res_dec', type=bool, default=False)

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

def evaluation(pred, gt, snr_idx, snr_max_idx, intensity_idx, intensity_max_idx, threshold_prob, threshold_trigger, sample_tolerant, mode, isREDPAN_dataset=False):
    tp, fp, tn, fn = 0, 0, 0, 0 
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

        if isREDPAN_dataset:
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

        if mode == 'single':
            a = np.where(pred[i] >= threshold_prob, 1, 0)

            if np.any(a):
                c = np.where(a==1)
                pred_isTrigger = True
                pred_trigger = c[0][0]
            else:
                pred_trigger = 0
        
        elif mode == 'avg':
            a = pd.Series(pred[i])  
            win_avg = a.rolling(window=threshold_trigger).mean().to_numpy()

            c = np.where(win_avg >= threshold_prob, 1, 0)

            pred_trigger = 0
            if c.any():
                tri = np.where(c==1)
                # pred_trigger = tri[0][0]-threshold_trigger+1
                pred_trigger = tri[0][0]
                pred_isTrigger = True
                
        elif mode == 'continue':
            tmp = np.where(pred[i] >= threshold_prob, 1, 0)
            
            a = pd.Series(tmp)    
            data = a.groupby(a.eq(0).cumsum()).cumsum().tolist()
            pred_trigger = 0
            if threshold_trigger in data:
                # pred_trigger = data.index(threshold_trigger)-threshold_trigger+1
                pred_trigger = data.index(threshold_trigger)
                pred_isTrigger = True
            else:
                pred_trigger = 0

        elif mode == 'max':
            pred_trigger = np.argmax(pred[i]).item()
            
            if pred[i][pred_trigger] >= threshold_prob:
                pred_isTrigger = True
            else:
                pred_trigger = 0

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

    return tp, fp, tn, fn, diff, abs_diff, res, snr_stat, intensity_stat, case_stat

def set_generators(opt, ptime=None):
    cwbsn, tsmip, stead, cwbsn_noise = load_dataset(opt)

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
    phase_dict = ['trace_p_arrival_sample']
    if not opt.allTest:
        ptime = opt.p_timestep  
    augmentations = basic_augmentations(opt, phase_dict=phase_dict, EEW=opt.EEW, test=True, ptime=ptime)
    
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

def inference(opt, model, test_loader, device):
    # 先把整個 test set 的預測結果都跑完一遍

    pred = []
    gt = []
    snr_total = []
    intensity_total = []
    mag, dis = [], []
    isREDPAN = True if opt.model_opt == 'RED_PAN' else False

    model.eval()
    with tqdm(test_loader) as epoch:
        for data in epoch:          
            if not opt.dataset_opt == 'REDPAN_dataset':
                snr_total += calc_snr(data, isREDPAN)
                intensity_total += calc_inten(data, isREDPAN)

            with torch.no_grad():
                if opt.dataset_opt == 'REDPAN_dataset':
                    if opt.model_opt == 'RED_PAN':
                        wf, psn, mask = data
                        out_PS, out_M = model(wf.to(device))
                        
                        for i in range(wf.shape[0]):
                            plt.subplot(311)
                            plt.plot(wf[i].T)
                            plt.subplot(312)
                            plt.plot(out_PS[i, 0].detach().cpu().numpy())
                            plt.subplot(313)
                            plt.plot(psn[i, 0].numpy())
                            plt.savefig(f'./tmp/{i}.png')
                            plt.clf()


                        pred += [out_PS[i, 0].detach().squeeze().cpu().numpy() for i in range(out_PS.shape[0])]
                        gt += [psn[i, 0].numpy() for i in range(wf.shape[0])]
                        break
                    elif opt.model_opt == 'conformer':
                        wf, psn, mask = data
                        out = model(wf.to(device))

                        if opt.label_type == 'other':
                            pred += [out[i, :, 0].detach().squeeze().cpu().numpy() for i in range(out.shape[0])]
                            gt += [psn[i, 0] for i in range(wf.shape[0])]
                        elif opt.label_type == 'p':
                            pred += [out[i].detach().squeeze().cpu().numpy() for i in range(out.shape[0])]
                            gt += [psn[i, 0] for i in range(wf.shape[0])]
                    elif opt.model_opt == 'GRADUATE':
                        wf, psn, mask, stft, seg = data
                        out_seg, out = model(wf.to(device), stft=stft.float().to(device))

                        if opt.label_type == 'other':
                            pred += [out[i, :, 0].detach().squeeze().cpu().numpy() for i in range(out.shape[0])]
                            gt += [psn[i, 0] for i in range(wf.shape[0])]
                        elif opt.label_type == 'p':
                            pred += [out[i].detach().squeeze().cpu().numpy() for i in range(out.shape[0])]
                            gt += [psn[i, 0] for i in range(wf.shape[0])]
                    else:
                        wf, psn, mask = data
                        out = model(wf.float().to(device))

                        out = out[1].detach().squeeze().cpu().numpy()
                       
                        pred += [out[i] for i in range(out.shape[0])]
                        gt += [psn[i, 0].numpy() for i in range(wf.shape[0])]
                        
                else:
                    if opt.model_opt == 'basicphaseAE':
                        out = sliding_prediction(opt, model, data)
                        target = data['y'][:, 0].squeeze().numpy()

                        pred += out
                        gt += [target[i] for i in range(len(target))]
                    elif opt.model_opt == 'RED_PAN':
                        out_PS, out_M = model(data['X'][:, :3].to(device))
                        target = data['X'][:, 3].squeeze().numpy()

                        pred += [out_PS[i, 0].detach().squeeze().cpu().numpy() for i in range(out_PS.shape[0])]
                        gt += [target[i] for i in range(target.shape[0])]
                    else:
                        if opt.model_opt == 'conformer_intensity':
                            out, out_MT = model(data['X'].to(device))
                        elif opt.model_opt == 'conformer_stft':
                            out = model(data['X'].to(device), stft=data['stft'].to(device).float())
                        elif opt.model_opt == 'GRADUATE':
                            _, out = model(data['X'].to(device), stft=data['stft'].to(device).float())
                        elif opt.model_opt == 'GRADUATE_MAG' or opt.model_opt == 'GRADUATE_MAG_noNorm':
                            _, out, out_mag = model(data['X'].to(device), stft=data['stft'].float().to(device), fft=data['fft'].float().to(device))
                        elif opt.model_opt == 'GRADUATE_MAG_deStationary':
                            _, out, out_mag = model(data['X'].to(device), stft=data['stft'].float().to(device), fft=data['fft'].float().to(device), mean_std=data['mean_std'].float().to(device))
                        elif opt.model_opt == 'GRADUATE_MAG24':
                            wf = torch.cat((data['X'], data['ori_X']), dim=1)
                            _, out, out_mag = model(wf.to(device), stft=data['stft'].float().to(device), fft=data['fft'].float().to(device))
                        elif opt.model_opt == 'tsfc':
                            _, out_mag, out = model(data['X'][:, :3].to(device), stft=data['stft'].float().to(device))
                        elif opt.model_opt == 'real_GRADUATE':
                            _, out_mag, out = model(data['X'].to(device), stft=data['stft'].float().to(device), fft=data['fft'].float().to(device), mean_std=data['mean_std'].float().to(device))
                        else:
                            out = model(data['X'].to(device))

                        if opt.model_opt == 'eqt':
                            if not opt.s_test:
                                out = out[1].detach().squeeze().cpu().numpy()
                            else:
                                out = out[2].detach().squeeze().cpu().numpy()
                        elif opt.model_opt == 'phaseNet':
                            out = out[:, 0].detach().squeeze().cpu().numpy()
                        elif opt.model_opt == 'tsfc':
                            out = out[1].detach().squeeze().cpu().numpy()

                            gt_mag = torch.max(data['mag'], dim=1).values
                            for i in range(out_mag.shape[0]):
                                mag.append((gt_mag[i], out_mag[i, 0, 0]))
                                dis.append(data['dis'][i])
                        elif opt.model_opt == 'real_GRADUATE':
                            out = out[:, :, 1].detach().squeeze().cpu().numpy()

                            gt_mag = torch.max(data['mag'], dim=1).values
                            for i in range(out_mag.shape[0]):
                                mag.append((gt_mag[i], out_mag[i, 0, 0]))
                                dis.append(data['dis'][i].item())
                        elif opt.model_opt == 'GRADUATE_MAG' or opt.model_opt == 'GRADUATE_MAG_noNorm' or opt.model_opt == 'GRADUATE_MAG24' or opt.model_opt == 'GRADUATE_MAG_deStationary':
                            if opt.label_type == 'p':
                                out = out.detach().squeeze().cpu().numpy()
                            elif opt.label_type == 'other':
                                out = out[:, :, 0].detach().squeeze().cpu().numpy()      
                            else:
                                out = out[1].detach().squeeze().cpu().numpy()

                            gt_mag = torch.max(data['mag'], dim=1).values
                            for i in range(out_mag.shape[0]):
                                mag.append((gt_mag[i], out_mag[i, 0, 0]))
                                dis.append(data['dis'][i].item())
                        else:
                            if opt.label_type == 'p':
                                out = out.detach().squeeze().cpu().numpy()
                            elif opt.label_type == 'other':
                                out = out[:, :, 0].detach().squeeze().cpu().numpy()                
                            elif opt.label_type == 'all':
                                if not opt.s_test:
                                    out = out[1].squeeze().detach().cpu().numpy()
                                else:
                                    out = out[2].squeeze().detach().cpu().numpy()

                        # plt.subplot(311)
                        # plt.plot(data['X'][0, :3].T)
                        # plt.subplot(312)
                        # plt.plot(data['y'][0].T)
                        # plt.subplot(313)
                        # plt.plot(out[0])
                        # plt.savefig(f"./tmp/{idx}.png")
                        # plt.clf()

                        if not opt.s_test:
                            target = data['y'][:, 0].squeeze().numpy()
                        else:
                            target = data['y'][:, 1].squeeze().numpy()
                        
                        if type(out) == list:
                            pass
                        elif out.ndim == 2:
                            pred += [out[i] for i in range(out.shape[0])]
                            gt += [target[i] for i in range(target.shape[0])]
                        else:
                            pred += [out]
                            gt += [target]
    
    return pred, gt, snr_total, intensity_total, mag, dis

def score(pred, gt, snr_total, intensity_total, mode, opt, threshold_prob, threshold_trigger, isTest=False):
    # 依照 snr 不同分別計算數據，先將原本的 snr level 轉換成對應 index
    # snr_level = list(np.arange(0.0, 3.5, 0.25)) + list(np.arange(3.5, 5.5, 0.5))
    snr_level = [-9999] + list(np.arange(-1.0, 0.0, 0.5)) + list(np.arange(0.0, 3.5, 0.25)) + list(np.arange(3.5, 5.5, 0.5))
    intensity_level = [-1, 0, 1, 2, 3, 4, 5, 5.5, 6, 6.5, 7]

    if not opt.dataset_opt == 'REDPAN_dataset':    
        snr_idx = convert_snr_to_level(snr_level, snr_total)
        intensity_idx = convert_intensity_to_level(intensity_level, intensity_total)
        
    if not opt.dataset_opt == 'REDPAN_dataset':
        tp, fp, tn, fn, diff, abs_diff, res, snr_stat, intensity_stat, case_stat = evaluation(pred, gt, snr_idx, len(snr_level), intensity_idx, len(intensity_level), threshold_prob, threshold_trigger, opt.sample_tolerant, mode)
    else:
        snr_idx, intensity_idx = 0, 0

        tp, fp, tn, fn, diff, abs_diff, res, snr_stat, intensity_stat, case_stat = evaluation(pred, gt, snr_idx, len(snr_level), intensity_idx, len(intensity_level), threshold_prob, threshold_trigger, opt.sample_tolerant, mode, isREDPAN_dataset=True)

    # print('tp=%d, fp=%d, tn=%d, fn=%d' %(tp, fp, tn, fn))

    # statisical  
    precision = tp / (tp+fp) if (tp+fp) != 0 else 0
    recall = tp / (tp+fn) if (tp+fn) != 0 else 0
    fpr = fp / (tn+fp) if (tn+fp) != 0 else 100
    fscore = 2*precision*recall / (precision+recall) if (precision+recall) != 0 else 0
    # mcc = (tp*tn-fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    logging.info('======================================================')
    logging.info('threshold_prob: %.2f' %(threshold_prob))
    logging.info('threshold_trigger: %d' %(threshold_trigger))
    logging.info('TPR=%.4f, FPR=%.4f, Precision=%.4f, Fscore=%.4f' %(recall, fpr, precision, fscore))
    logging.info('tp=%d, fp=%d, tn=%d, fn=%d' %(tp, fp, tn, fn))
    logging.info('abs_diff=%.4f, diff=%.4f' %(np.mean(abs_diff)/100, np.mean(diff)/100))
    logging.info('trigger_mean=%.4f, trigger_std=%.4f' %(np.mean(diff)/100, np.std(diff)/100))
    # logging.info('MCC=%.4f' %(mcc))
    logging.info('RMSE=%.4f, MAE=%.4f' %(np.sqrt(np.mean(np.array(diff)**2))/100, np.mean(abs_diff)/100))

    if isTest:
        toLine(opt.save_path, precision, recall, fscore, np.mean(diff)/100, np.std(diff)/100)

    return fscore, abs_diff, diff, snr_stat, intensity_stat, case_stat

def mag_score(mag, dis):
    abs_diff, diff = 0.0, 0.0
    distance_level = [20, 50, 80, 100, 150, 200]

    pred, gt = [], []
    dis_mag_stat = {}
    for i in range(len(distance_level)):
        dis_mag_stat[i] = []

    print('Calculating mag score...')
    for i in tqdm(range(len(mag)), total=len(mag)):
        dis_idx = bisect.bisect_right(distance_level, dis[i])-1
        if dis_idx < 0:
            dis_idx = 0

        # prediction - ground_truth
        mag_diff = mag[i][1] - mag[i][0]

        abs_diff += abs(mag_diff)
        diff += mag_diff

        dis_mag_stat[dis_idx].append(mag_diff.item())

        pred.append(mag[i][1].detach().cpu().numpy())
        gt.append(mag[i][0].numpy())
    
    return abs_diff / len(mag), diff / len(mag), pred, gt, dis_mag_stat

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
        output_dir = os.path.join(output_dir, level)
    else:
        output_dir = os.path.join(output_dir, f"allTest_{opt.dataset_opt}")
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    subpath = 'threshold'
    if opt.level != -1:
        subpath = subpath + '_' + str(opt.level)
    if opt.p_timestep != 750:
        subpath = subpath + '_' + str(opt.p_timestep)
    if opt.allTest:
        subpath = subpath + '_allCase_testing_' + str(opt.level)
    if opt.load_specific_model != 'None':
        subpath = subpath + '_' + opt.load_specific_model
    if opt.s_test:
        subpath = subpath + '_Swave'

    subpath = subpath + '.log'
    print('logpath: ', subpath)
    log_path = os.path.join(output_dir, subpath)

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        filename=log_path, 
                        filemode='a', 
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S",)
    print(opt.save_path)

    # 設定 device (opt.device = 'cpu' or 'cuda:X')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load datasets
    if not opt.allTest:
        print('loading datasets')
        if not opt.dataset_opt == 'REDPAN_dataset':
            dev_generator, test_generator = set_generators(opt)
            dev_loader = DataLoader(dev_generator, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
            test_loader = DataLoader(test_generator, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
        else:
            basedir = '/mnt/disk4/weiwei/seismic_datasets/REDPAN_30S_pt/'
            if opt.model_opt == 'RED_PAN' or opt.model_opt == 'eqt':
                dev_set = REDPAN_dataset(basedir, 'val', 1.0, opt.model_opt, load_to_ram=opt.load_to_ram)
                test_set = REDPAN_dataset(basedir, 'test', 1.0, opt.model_opt, load_to_ram=opt.load_to_ram)
            else:
                dev_set, test_set = REDPAN_dataset(basedir, 'val', 1.0, 'REDPAN'), REDPAN_dataset(basedir, 'test', 1.0, 'REDPAN')
            
            # create dataloaders
            print('creating dataloaders')
            dev_loader = DataLoader(dev_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
            test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    # load model
    model = load_model(opt, device)

    if opt.save_path != 'eqt_original':
        if opt.load_specific_model != 'None':
            print('loading ', opt.load_specific_model)
            model_path = os.path.join(model_dir, opt.load_specific_model+'.pt')
        elif not opt.load_last:
            print('loading best checkpoint')
            model_path = os.path.join(model_dir, 'model.pt')
        else:
            print('loading last checkpoint')
            model_path = os.path.join(model_dir, 'checkpoint_last.pt')

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model_path = os.path.join(model_dir, 'model.pt')
        checkpoint = torch.load("/mnt/disk4/weiwei/newstead11.pt", map_location=device)
        model = load_my_state_dict(model, checkpoint)

    # start finding
    max_fscore = 0.0
    cnt = 0
    front = opt.sample_tolerant
    back = opt.sample_tolerant

    if opt.threshold_type == 'all':
        mode = ['max', 'single', 'continue', 'avg']  # avg, continue
        # mode = ['single', 'continue', 'avg']
    elif opt.threshold_type == 'avg':
        mode = ['avg']
    elif opt.threshold_type == 'continue':
        mode = ['continue']
    elif opt.threshold_type == 'single':
        mode = ['single']
    elif opt.threshold_type == 'max':
        mode = ['max']

    if not opt.do_test and not opt.allTest:
        # find the best criteria
        print('finding best criteria...')
        pred, gt, snr_total, intensity_total, mag, dis = inference(opt, model, dev_loader, device)

        if opt.model_opt == 'real_GRADUATE' or opt.model_opt == 'GRADUATE_MAG' or opt.model_opt == 'GRADUATE_MAG_noNorm' \
            or opt.model_opt == 'GRADUATE_MAG24' or opt.model_opt == 'GRADUATE_MAG_deStationary':
            mag_abs_diff, mag_diff, mag_pred, mag_gt, dis_mag = mag_score(mag, dis)

            print(f"Magnitude estimation -> abs_diff: {mag_abs_diff}, diff: {mag_diff}")

            with open(os.path.join(output_dir, 'mag_pred.pkl'), 'wb') as f:
                pickle.dump(mag_pred, f)
            with open(os.path.join(output_dir, 'mag_gt.pkl'), 'wb') as f:
                pickle.dump(mag_gt, f)
            with open(os.path.join(output_dir, 'snr_total.pkl'), 'wb') as f:
                pickle.dump(snr_total, f)
            with open(os.path.join(output_dir, 'dismag.json'), 'w') as f:
                json.dump(dis_mag, f)

        best_fscore = 0.0
        best_mode = ""
        best_prob = 0.0
        best_trigger = 0
        for m in mode:
            logging.info('======================================================')
            logging.info('Mode: %s' %(m))

            for prob in tqdm(np.arange(opt.threshold_prob_start, opt.threshold_prob_end, 0.05)):  # (0.45, 0.85)
                max_fscore = 0.0
                cnt = 0

                for trigger in np.arange(opt.threshold_trigger_start, opt.threshold_trigger_end, 5): # (10, 55)
                    fscore, abs_diff, diff, snr_stat, intensity_stat, case_stat = score(pred, gt, snr_total, intensity_total, m, opt, prob, trigger)
                    print('prob: %.2f, trigger: %d, fscore: %.4f' %(prob, trigger, fscore))

                    if fscore > max_fscore:
                        max_fscore = fscore
                        cnt = 0

                    else:
                        cnt += 1

                    if cnt == 1 or fscore == 0.0:
                        break

                    if fscore > best_fscore:
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
                        
                        best_fscore = fscore
                        best_mode = m
                        best_prob = prob
                        best_trigger = trigger

                    if m == 'single' or m == 'max':
                        break
            
        logging.info('======================================================')
        logging.info("Best: ")
        logging.info(f"mode: {best_mode}, prob: {best_prob}, trigger: {best_trigger}, fscore: {best_fscore}")
        if opt.model_opt == 'real_GRADUATE':
            logging.info(f"Magnitude estimation -> abs_diff: {mag_abs_diff}, diff: {mag_diff}")
        logging.info('======================================================')

    if opt.do_test or opt.dataset_opt == 'stead' or opt.dataset_opt == 'REDPAN_dataset':
        if opt.do_test:
            best_mode = opt.threshold_type
            best_prob = opt.threshold_prob_start
            best_trigger = opt.threshold_trigger_start

        logging.info('Inference on testing set')
        pred, gt, snr_total, intensity_total, mag, dis = inference(opt, model, test_loader, device)
        if opt.model_opt == 'real_GRADUATE' or opt.model_opt == 'GRADUATE_MAG' or opt.model_opt == 'GRADUATE_MAG_noNorm' \
            or opt.model_opt == 'GRADUATE_MAG24' or opt.model_opt == 'GRADUATE_MAG_deStationary':
            mag_abs_diff, mag_diff, mag_pred, mag_gt, dis_mag = mag_score(mag, dis)

            print(f"Magnitude estimation -> abs_diff: {mag_abs_diff}, diff: {mag_diff}")

            with open(os.path.join(output_dir, 'test_mag_pred.pkl'), 'wb') as f:
                pickle.dump(mag_pred, f)
            with open(os.path.join(output_dir, 'test_mag_gt.pkl'), 'wb') as f:
                pickle.dump(mag_gt, f)
            with open(os.path.join(output_dir, 'test_snr_total.pkl'), 'wb') as f:
                pickle.dump(snr_total, f)
            with open(os.path.join(output_dir, 'test_dismag.json'), 'w') as f:
                json.dump(dis_mag, f)

        fscore, abs_diff, diff, snr_stat, intensity_stat, case_stat = score(pred, gt, snr_total, intensity_total, best_mode, opt, best_prob, best_trigger, True)
        print('fscore: %.4f' %(fscore))
        if opt.model_opt == 'real_GRADUATE' or opt.model_opt == 'GRADUATE_MAG' or opt.model_opt == 'GRADUATE_MAG_noNorm' \
            or opt.model_opt == 'GRADUATE_MAG24' or opt.model_opt == 'GRADUATE_MAG_deStationary':
            logging.info(f"Magnitude estimation -> abs_diff: {mag_abs_diff}, diff: {mag_diff}")

        with open(os.path.join(output_dir, 'test_abs_diff_'+str(opt.level)+'.pkl'), 'wb') as f:
            pickle.dump(abs_diff, f)

        with open(os.path.join(output_dir, 'test_diff_'+str(opt.level)+'.pkl'), 'wb') as f:
            pickle.dump(diff, f)

        with open(os.path.join(output_dir, 'test_snr_stat_'+str(opt.level)+'.json'), 'w') as f:
            json.dump(snr_stat, f)

        with open(os.path.join(output_dir, 'test_intensity_stat_'+str(opt.level)+'.json'), 'w') as f:
            json.dump(intensity_stat, f)

        with open(os.path.join(output_dir, 'test_case_stat_'+str(opt.level)+'.json'), 'w') as f:
            json.dump(case_stat, f)

    # 將 p arrival 固定在多個不同時間點，分別得到實驗結果
    if opt.allTest:
        logging.info('configs: ')
        logging.info(opt)
        logging.info('dataset: %s' %(opt.dataset_opt))
        
        print('Start testing...')
        if opt.EEW_allTest:
            ptime_list = [2500, 2750]
        else:
            ptime_list = [750, 1500, 2000, 2500, 2750]
        
        best_mode = opt.threshold_type
        best_prob = opt.threshold_prob_start
        best_trigger = opt.threshold_trigger_start

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
            pred, gt, snr_total, intensity_total, mag, dis = inference(opt, model, test_loader, device)
            if opt.model_opt == 'real_GRADUATE' or opt.model_opt == 'GRADUATE_MAG' or opt.model_opt == 'GRADUATE_MAG_noNorm' \
                or opt.model_opt == 'GRADUATE_MAG24' or opt.model_opt == 'GRADUATE_MAG_deStationary':
                mag_abs_diff, mag_diff, mag_pred, mag_gt, dis_mag = mag_score(mag, dis)
                
                logging.info(f"Magnitude estimation -> abs_diff: {mag_abs_diff}, diff: {mag_diff}")

                with open(os.path.join(new_output_dir, 'test_mag_pred.pkl'), 'wb') as f:
                    pickle.dump(mag_pred, f)
                with open(os.path.join(new_output_dir, 'test_mag_gt.pkl'), 'wb') as f:
                    pickle.dump(mag_gt, f)
                with open(os.path.join(new_output_dir, 'test_snr_total.pkl'), 'wb') as f:
                    pickle.dump(snr_total, f)
                with open(os.path.join(new_output_dir, 'test_dismag.json'), 'w') as f:
                    json.dump(dis_mag, f)

            fscore, abs_diff, diff, snr_stat, intensity_stat, case_stat = score(pred, gt, snr_total, intensity_total, best_mode, opt, best_prob, best_trigger, True)
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
