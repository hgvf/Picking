import numpy as np
import os
import argparse
import pandas as pd
import math
import logging
import json
import bisect
import requests
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

from calc import calc_intensity
from snr import snr_p
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import seisbench.data as sbd
import seisbench.generate as sbg

import sys
sys.path.append('/mnt/disk4/weiwei/continuous_picking/')
from trigger import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--criteria', type=str, default='median')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--save_path", type=str, default='tmp')
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--only_false', type=bool, default=False)
    parser.add_argument('--only_true', type=bool, default=False)
    parser.add_argument('--only_fp', type=bool, default=False)
    parser.add_argument('--only_fn', type=bool, default=False)
    parser.add_argument('--n_plot', type=int, default=40)
    parser.add_argument('--snr_condition', type=bool, default=False)
    parser.add_argument('--snr_lowerbound', type=float, default=4.5)
    parser.add_argument('--snr_upperbound', type=float, default=50)
    parser.add_argument('--intensity_condition', type=bool, default=False)
    parser.add_argument('--intensity_lowerbound', type=float, default=0)
    parser.add_argument('--intensity_upperbound', type=float, default=7)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--p_timestep', type=int, default=750)
    parser.add_argument('--window_size', type=int, default=3000)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--snr_threshold', type=float, default=-1)
    
    # dataset hyperparameters
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--level', type=int, default=-1)
    parser.add_argument('--s_wave', type=bool, default=False)

    # threshold
    parser.add_argument('--threshold_trigger', type=int, default=40)
    parser.add_argument('--sample_tolerant', type=int, default=50)
    parser.add_argument('--threshold_prob', type=float, default=0.55)
    parser.add_argument('--threshold_type', type=str, default='avg')

    # seisbench options
    parser.add_argument('--model_opt', type=str, default='none')
    parser.add_argument('--loss_weight', type=float, default=50)
    parser.add_argument('--dataset_opt', type=str, default='taiwan')
    parser.add_argument('--loading_method', type=str, default='full')
    parser.add_argument('--normalize_opt', type=str, default='peak')

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
    parser.add_argument('--emb_type', type=str, default='transformer_MLM')
    parser.add_argument('--emb_d_ffn', type=int, default=256)
    parser.add_argument('--emb_layers', type=int, default=4)
    parser.add_argument('--pretrained_emb', type=str)
    parser.add_argument('--dim_spectrogram', type=str, default='1D')
    parser.add_argument('--MGAN_normtype', type=str, default='mean')
    parser.add_argument('--MGAN_l', type=int, default=10)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_class', type=int, default=128)
    parser.add_argument('--query_type', type=str, default='pos_emb')
    parser.add_argument('--intensity_MT', type=bool, default=False)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # GRADUATE model
    parser.add_argument('--cross_attn_type', type=int, default=1)
    parser.add_argument('--n_segmentation', type=int, default=5)
    parser.add_argument('--output_layer_type', type=str, default='fc')
    parser.add_argument('--rep_KV', type=str, default='False')
    parser.add_argument('--segmentation_ratio', type=float, default=0.3)
    parser.add_argument('--seg_proj_type', type=str, default='crossattn')

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

def evaluation(pred, gt, snr_idx, snr_max_idx, intensity_idx, intensity_max_idx, threshold_prob, threshold_trigger, sample_tolerant, mode):
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
        snr_cur = snr_idx[i]
        intensity_cur = intensity_idx[i]

        c = np.where(gt[i] == 1)

        if len(c[0]) > 0:
            gt_isTrigger = True            
            gt_trigger = c[0][0]

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
                pred_trigger = tri[0][0]-threshold_trigger+1
                pred_isTrigger = True
                
        elif mode == 'continue':
            tmp = np.where(pred[i] >= threshold_prob, 1, 0)
            
            a = pd.Series(tmp)    
            data = a.groupby(a.eq(0).cumsum()).cumsum().tolist()

            if threshold_trigger in data:
                pred_trigger = data.index(threshold_trigger)-threshold_trigger+1
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

def set_generators(opt):
    cwbsn, tsmip, stead, cwbsn_noise = load_dataset(opt)

    # split datasets
    if opt.dataset_opt == 'all':
        cwbsn_test = cwbsn.test()
        tsmip_test = tsmip.test()
        cwbsn_noise_test = cwbsn_noise.test()
        
        test = cwbsn_test + tsmip_test + cwbsn_noise_test
    elif opt.dataset_opt == 'cwbsn':
        test = cwbsn.test()
    elif opt.dataset_opt == 'tsmip':
        test = tsmip.test()
    elif opt.dataset_opt == 'stead':
        _, dev, test = stead.train_dev_test()
    elif opt.dataset_opt == 'redpan' or opt.dataset_opt == 'taiwan':
        cwbsn_test = cwbsn.test()
        tsmip_test = tsmip.test()
        cwbsn_noise_test = cwbsn_noise.test()
        
        test = cwbsn_test + tsmip_test + cwbsn_noise_test
    elif opt.dataset_opt == 'prev_taiwan':
        cwbsn_test = cwbsn.test()
        tsmip_test = tsmip.test()
        stead_test = stead.test()
        
        test = cwbsn_test + tsmip_test + stead_test

    print(f'total traces -> test: {len(test)}')

    test_generator = sbg.GenericGenerator(test)

    # set generator with or without augmentations
    phase_dict = ['trace_p_arrival_sample']
    if opt.model_opt == 'conformer':
        augmentations = [
            sbg.VtoA(),
            sbg.Filter(N=5, Wn=[1, 45], btype='bandpass', keep_ori=True),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
            sbg.CharStaLta(),
            sbg.ChangeDtype(np.float32),
            sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),]
    elif opt.model_opt == 'RED_PAN':
        augmentations = [
            sbg.VtoA(),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
            sbg.ChangeDtype(np.float32),
            sbg.RED_PAN_label(),]
    elif opt.model_opt == 'eqt':
        augmentations = [
            sbg.VtoA(),
            sbg.Filter(N=5, Wn=[1, 45], btype='bandpass', keep_ori=True),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
            sbg.ChangeDtype(np.float32),
            sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0, shape='triangle'),
            ]

    test_generator.add_augmentations(augmentations)

    return test_generator

def sliding_prediction(model, wave, window_size=3000, step_size=100, device='cpu', criteria='median'):
    # wf: (batch, 12, wave_length)
    wave_length = wave.shape[-1]
    batch = wave.shape[0]

    # create empty list to store the prediction
    total_pred = []
    res = []
    for j in range(batch):
        pred = []
        for i in range(wave_length):
            pred.append([])
        total_pred.append(pred)
        res.append([])

    # start predicting
    with torch.no_grad():
        for i in range(0, wave_length-window_size, step_size):
            if opt.model_opt == 'conformer':
                out = model(wave[:, :, i:i+window_size].to(device), window_size).detach().cpu().numpy()
            elif opt.model_opt == 'eqt':
                out = model(wave[:, :, i:i+window_size].to(device))[1].detach().cpu().numpy()              
            elif opt.model_opt == 'RED_PAN':
                out_PS, out_M = model(wave[:, :, i:i+window_size].to(device))
                out = out_PS[:, 0].detach().cpu().numpy()         
            
            for b in range(batch):
                for j in range(window_size):
                    total_pred[b][i+j].append(out[b][j])
    
    for b in range(batch):
        for i in range(len(pred)):
            if len(total_pred[b][i]) == 0:
                continue
            
            # 取中位數
            if criteria == 'median':
                res[b].append(np.median(total_pred[b][i]))
            # 把前後幾個 samples 去除掉，只保留中段後再取中位數
            elif criteria == 'middle':
                n = len(total_pred[b][i])
                print('n: ', n)
                print('remaining: ', len(total_pred[b][i][int(round((n//2)*0.33)):int(round(-(n//2)*0.33))]))
                res[b].append(np.median(total_pred[b][i][int(round((n//2)*0.33)):int(round(-(n//2)*0.33))]))
            
    return np.array(res)

def plot(wave, pred, gt, res, step, pred_trigger, gt_trigger, snr_cur, intensity_cur, plot_path):
    wave = wave.cpu().numpy()

    plt.figure(figsize=(18, 25))
    plt.rcParams.update({'font.size': 18})
    plt.subplot(5,1,1)

    plt.subplot(511)
    plt.plot(wave[0, 0])
    if snr_cur != 'noise':
        plt.axvline(x=gt_trigger, color='y', label='labeled')
    if pred_trigger != 0:
        plt.axvline(x=pred_trigger, color='r', label='predicted')
    plt.title('Z')

    plt.subplot(512)
    plt.plot(wave[0, 1])
    if snr_cur != 'noise':
        plt.axvline(x=gt_trigger, color='y', label='labeled')
    if pred_trigger != 0:
        plt.axvline(x=pred_trigger, color='r', label='predicted')
    plt.title('N')

    plt.subplot(513)
    plt.plot(wave[0, 2])
    if snr_cur != 'noise':
        plt.axvline(x=gt_trigger, color='y', label='labeled')
    if pred_trigger != 0:
        plt.axvline(x=pred_trigger, color='r', label='predicted')
    plt.title('E')

    plt.subplot(514)
    plt.plot(pred)
    if snr_cur != 'noise':
        plt.axvline(x=gt_trigger, color='y', label='labeled')
    if pred_trigger != 0:
        plt.axvline(x=pred_trigger, color='r', label='predicted')
    plt.ylim([-0.05, 1.05])
    pred_title = 'pred (' + str(pred_trigger) + ')'
    plt.title(pred_title)

    plt.subplot(515)
    plt.plot(gt)
    if snr_cur != 'noise':
        plt.axvline(x=gt_trigger, color='y', label='labeled')
    if pred_trigger != 0:
        plt.axvline(x=pred_trigger, color='r', label='predicted')
    plt.ylim([-0.05, 1.05])
    gt_title = 'ground truth (' + str(gt_trigger) + ')'
    plt.title(gt_title)
    plt.legend()

    if snr_cur == 'noise':
        filename = res + '_' + '_' + str(step) + '_'  + str(snr_cur) + '_' + str(pred_trigger) + '_' + str(gt_trigger) + '.png'
    else:
        filename = res + '_' + '_' + str(step) + '_' + str(intensity_cur) + '_' + str(round(snr_cur, 4)) + '_' + str(pred_trigger) + '_' + str(gt_trigger) + '.png'

    png_path = os.path.join(plot_path, filename)
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

def collate_fn(data):
    max_length = max([data[i]['X'].shape[-1] for i in range(len(data))])

    newdata = {}
    wave, ori_wave, y = [], [], []
    for i in range(len(data)):
        cur_length = data[i]['X'].shape[-1]

        if max_length - cur_length > 0:
            wave.append(torch.cat((torch.FloatTensor(data[i]['X']), torch.ones(data[i]['X'].shape[0], max_length-cur_length)), dim=-1))
            ori_wave.append(torch.cat((torch.FloatTensor(data[i]['ori_X']), torch.ones(data[i]['ori_X'].shape[0], max_length-cur_length)), dim=-1))
        else:
            wave.append(torch.FloatTensor(data[i]['X']))
            ori_wave.append(torch.FloatTensor(data[i]['ori_X']))

        if 'y' in data[0].keys():
            if max_length - cur_length > 0:
                y.append(torch.cat((torch.FloatTensor(data[i]['y']), torch.ones(data[i]['y'].shape[0], max_length-cur_length)), dim=-1))
            else:
                y.append(torch.FloatTensor(data[i]['y']))
    
    newdata['X'] = torch.stack(wave)
    newdata['ori_X'] = torch.stack(ori_wave)

    if 'y' in data[0].keys():
        newdata['y'] = torch.stack(y)

    return newdata

if __name__ == '__main__':
    matplotlib.use('Agg')   
    opt = parse_args()

    plot_path = os.path.join('./plot/' + opt.save_path, 'whole_testing')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    output_dir = os.path.join('./results/', opt.save_path)
    model_dir = output_dir
    if opt.level == -1:
        level = 'all'
    else:
        level = str(opt.level)

    output_dir = os.path.join(output_dir, level)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    subpath = 'whole_test'
    if opt.level != -1:
        subpath = subpath + '_' + str(opt.level) + '_' + opt.criteria
    
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
    print('loading datasets')
    test_generator = set_generators(opt)
    test_loader = DataLoader(test_generator, batch_size=opt.batch_size, shuffle=False, num_workers=6, collate_fn=collate_fn)

    # load model
    model = load_model(opt, device)
    model_path = os.path.join(model_dir, 'model.pt')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    tp, fp, tn, fn = 0, 0, 0, 0 
    idx = 0
    with tqdm(test_loader) as epoch:
        for data in epoch:
            idx += 1
            
            # if not noise waveform, calculate log(SNR)
            gt = data['y'][0, 0] if not opt.model_opt == 'RED_PAN' else data['X'][0, 3]
            tri = torch.where(gt == 1)[0]
            if len(tri) != 0:
                if not opt.model_opt == 'RED_PAN':
                    snr_tmp = snr_p(data['ori_X'][0, 0, :].cpu().numpy(), data['y'][0, 0].cpu().numpy())
                else:
                    snr_tmp = snr_p(data['ori_X'][0, 0, :].cpu().numpy(), data['X'][0, 3].cpu().numpy())

                if snr_tmp is None:
                    snr_cur = 'noise'
                else:
                    snr_cur = np.log10(snr_tmp)
                    
            else:
                snr_cur = 'noise'

            if opt.snr_condition:
                if snr_cur == 'noise' or snr_cur >= opt.snr_upperbound or snr_cur <= opt.snr_lowerbound:
                    continue

            intensity_cur = calc_intensity(data['ori_X'][0, 0].numpy(), data['ori_X'][0, 1].numpy(), data['ori_X'][0, 2].numpy(), 'Acceleration', 100)
            
            if opt.intensity_condition:
                if snr_cur == 'noise' or intensity_tmp >= opt.intensity_upperbound or intensity_tmp <= opt.intensity_lowerbound:
                    continue

            if opt.model_opt == 'RED_PAN':
                x = data['X'][:, :3].to(device)
                gt = data['X'][:, 3]
                
            else:
                x = data['X']
                gt = data['y'][:, 0]

            out = sliding_prediction(model, x, window_size=opt.window_size, step_size=opt.step_size, device=device, criteria=opt.criteria)        
            
            trigger_args = (opt.threshold_prob, opt.threshold_trigger, opt.threshold_type)
            detrigger_args = (0.1, 50)
            tri = trigger(out, trigger_args, detrigger_args)
            
            res = judge(tri, [torch.argmax(gt[i]).item() for i in range(gt.shape[0])], opt.sample_tolerant, x.shape[-1])
            
            for b in range(x.shape[0]):
                if res[b][0] == 'tp':
                    tp += 1
                if res[b][0] == 'fp':
                    fp += 1
                if res[b][0] == 'tn':
                    tn += 1
                if res[b][0] == 'fn':
                    fn += 1    
            
            if (idx+1) % 500 == 0:
                print(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}")
            
            if opt.plot:
                # padded out
                diff = x.shape[-1] - out.shape[-1]
                if diff > 0:
                    out = np.concatenate((out, np.zeros((1, diff))), axis=-1)
                if (opt.only_true and (res[0][0] == 'tp' or res[0][0] == 'tn')) or (opt.only_false and (res[0][0] == 'fp' or res[0][0] == 'fn')) or (opt.only_fn and res[0][0] == 'fn') or (opt.only_fp and res[0][0] == 'fp') or (not opt.only_true and not opt.only_false and not opt.only_fp and not opt.only_fn):
                    plot(x, out[0], gt[0], res[0][0], idx, res[0][2], torch.argmax(gt).item(), snr_cur, intensity_cur, plot_path)

    # statisical  
    precision = tp / (tp+fp) if (tp+fp) != 0 else 0
    recall = tp / (tp+fn) if (tp+fn) != 0 else 0
    fpr = fp / (tn+fp) if (tn+fp) != 0 else 100
    fscore = 2*precision*recall / (precision+recall) if (precision+recall) != 0 else 0

    print(f"Recall: {round(recall, 4)}), precision: {round(precision, 4)}, fpr: {round(fpr, 4)}, fscore: {round(fscore, 4)}")
    print(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}")

    logging.info('======================================================')
    logging.info(f'threshold_type: {opt.threshold_type}')
    logging.info(f'threshold_prob: {opt.threshold_prob}')
    logging.info(f'threshold_trigger: {opt.threshold_trigger}')
    logging.info(f"Recall: {round(recall, 4)}), precision: {round(precision, 4)}, fpr: {round(fpr, 4)}, fscore: {round(fscore, 4)}")
    logging.info(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}")

    
