import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
from tqdm import tqdm

from snr import snr_p
from utils import *
from calc import calc_intensity

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import seisbench.data as sbd
import seisbench.generate as sbg

def parse_args():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--dataset_opt', type=str, default='all')
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
    parser.add_argument('--window_size', type=int, default=1000)
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

    opt = parser.parse_args()

    return opt

def plot(d, out, target, step, res, pred_trigger, gt_trigger, snr_cur, intensity_cur, plot_path, ori=False, zoomin=False, superzoomin=False):
    plt.figure(figsize=(18, 25))
    plt.rcParams.update({'font.size': 18})
    plt.subplot(5,1,1)

    if zoomin:
        if pred_trigger == 0:
            pred_trigger = gt_trigger

    if d.ndim == 3:
        if superzoomin:
            plt.plot(d[0, 0, pred_trigger-100:pred_trigger+100])
        else:
            plt.plot(d[0, 0, :]) if not zoomin else plt.plot(d[0, 0, pred_trigger-200:pred_trigger+200])
    else:
        if superzoomin:
            plt.plot(d[0, pred_trigger-100:pred_trigger+100])
        else:
            plt.plot(d[0, :]) if not zoomin else plt.plot(d[0, pred_trigger-200:pred_trigger+200])
    
    if not zoomin and not superzoomin:
        if res != 'fn':
            plt.axvline(x=pred_trigger, color='r', label='predicted')
        plt.axvline(x=gt_trigger, color='y', label='labeled')
    else:
        if pred_trigger != 0:
            if superzoomin:
                plt.axvline(x=100, color='r', label='predicted')
                plt.axvline(x=gt_trigger-(pred_trigger-100), color='y', label='labeled')
            else:
                plt.axvline(x=200, color='r', label='predicted')
                plt.axvline(x=gt_trigger-(pred_trigger-200), color='y', label='labeled')
        else:
            plt.axvline(x=pred_trigger, color='y', label='labeled')

    plt.title('Z')

    plt.subplot(5,1,2)
    if d.ndim == 3:
        if superzoomin:
            plt.plot(d[0, 1, pred_trigger-100:pred_trigger+100])
        else:
            plt.plot(d[0, 1, :]) if not zoomin else plt.plot(d[0, 1, pred_trigger-200:pred_trigger+200])
    else:
        if superzoomin:
            plt.plot(d[1, pred_trigger-100:pred_trigger+100])
        else:
            plt.plot(d[1, :]) if not zoomin else plt.plot(d[1, pred_trigger-200:pred_trigger+200])
    
    if not zoomin and not superzoomin:
        if res != 'fn':
            plt.axvline(x=pred_trigger, color='r', label='predicted')
        plt.axvline(x=gt_trigger, color='y', label='labeled')
    else:
        if pred_trigger != 0:
            if superzoomin:
                plt.axvline(x=100, color='r', label='predicted')
                plt.axvline(x=gt_trigger-(pred_trigger-100), color='y', label='labeled')
            else:
                plt.axvline(x=200, color='r', label='predicted')
                plt.axvline(x=gt_trigger-(pred_trigger-200), color='y', label='labeled')
        else:
            plt.axvline(x=pred_trigger, color='y', label='labeled')

    plt.title('N')

    plt.subplot(5,1,3)
    if d.ndim == 3:
        if superzoomin:
            plt.plot(d[0, 2, pred_trigger-100:pred_trigger+100])
        else:
            plt.plot(d[0, 2, :]) if not zoomin else plt.plot(d[0, 2, pred_trigger-200:pred_trigger+200])
    else:
        if superzoomin:
            plt.plot(d[2, pred_trigger-100:pred_trigger+100])
        else:
            plt.plot(d[2, :]) if not zoomin else plt.plot(d[2, pred_trigger-200:pred_trigger+200])
    
    if not zoomin and not superzoomin:
        if res != 'fn':
            plt.axvline(x=pred_trigger, color='r', label='predicted')
        plt.axvline(x=gt_trigger, color='y', label='labeled')
    else:
        if pred_trigger != 0:
            if superzoomin:
                plt.axvline(x=100, color='r', label='predicted')
                plt.axvline(x=gt_trigger-(pred_trigger-100), color='y', label='labeled')
            else:
                plt.axvline(x=200, color='r', label='predicted')
                plt.axvline(x=gt_trigger-(pred_trigger-200), color='y', label='labeled')
        else:
            plt.axvline(x=pred_trigger, color='y', label='labeled')
        
    plt.title('E')

    plt.subplot(5,1,4)
    if not zoomin and not superzoomin:
        plt.plot(out) 
        if res != 'fn':
            plt.axvline(x=pred_trigger, color='r', label='predicted')
        plt.axvline(x=gt_trigger, color='y', label='labeled')
    else:
        if superzoomin:
            plt.plot(out[pred_trigger-100:pred_trigger+100])
        else:
            plt.plot(out[pred_trigger-200:pred_trigger+200])
        if pred_trigger != 0:
            if superzoomin:
                plt.axvline(x=100, color='r', label='predicted')
                plt.axvline(x=gt_trigger-(pred_trigger-100), color='y', label='labeled')
            else:
                plt.axvline(x=200, color='r', label='predicted')
                plt.axvline(x=gt_trigger-(pred_trigger-200), color='y', label='labeled')
        else:
            plt.axvline(x=pred_trigger, color='y', label='labeled')
    plt.ylim([-0.05, 1.05])
    
    pred_title = 'pred (' + str(pred_trigger) + ')'
    plt.title(pred_title)

    plt.subplot(5,1,5)
    if not zoomin and not superzoomin:
        plt.plot(target)
        if res != 'fn':
            plt.axvline(x=pred_trigger, color='r', label='predicted')
        plt.axvline(x=gt_trigger, color='y', label='labeled')
    else:
        if superzoomin:
            plt.plot(target[pred_trigger-100:pred_trigger+100])
        else:
            plt.plot(target[pred_trigger-200:pred_trigger+200])
        if pred_trigger != 0:
            if superzoomin:
                plt.axvline(x=100, color='r', label='predicted')
                plt.axvline(x=gt_trigger-(pred_trigger-100), color='y', label='labeled')
            else:
                plt.axvline(x=200, color='r', label='predicted')
                plt.axvline(x=gt_trigger-(pred_trigger-200), color='y', label='labeled')
        else:
            plt.axvline(x=pred_trigger, color='y', label='labeled')
    plt.ylim([-0.05, 1.05])
    gt_title = 'ground truth (' + str(gt_trigger) + ')'
    plt.title(gt_title)
    plt.legend()

    if ori:
        ori = 'original'
    elif zoomin:
        ori = 'zoomin'
    elif superzoomin:
        ori = 'superzoomin'
    else:
        ori = 'zscored'

    if snr_cur == 'noise':
        filename = res + '_' + ori + '_' + str(step) + '_'  + str(snr_cur) + '_' + str(pred_trigger) + '_' + str(gt_trigger) + '.png'
    else:
        filename = res + '_' + ori + '_' + str(step) + '_' + str(intensity_cur) + '_' + str(round(snr_cur, 4)) + '_' + str(pred_trigger) + '_' + str(gt_trigger) + '.png'

    png_path = os.path.join(plot_path, filename)
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

def evaluation(pred, gt, threshold_prob, threshold_trigger, sample_tolerant, threshold_type):
    tp=fp=tn=fn=0
    pred_isTrigger = False
    gt_isTrigger = False
    diff = []
    abs_diff = []

    gt_trigger = 0
    if gt.any():
        c = np.where(gt == 1)

        if len(c[0]) > 0:
            gt_trigger = c[0][0]
            gt_isTrigger = True
        else:
            c = np.where(gt == 0.95)
          
            if len(c[0]) > 0:
                gt_trigger = c[0][0]
                gt_isTrigger = True

    if threshold_type == 'single':
        a = np.where(pred >= threshold_prob, 1, 0)

        if np.any(a):
            c = np.where(a==1)
            pred_isTrigger = True
            pred_trigger = c[0][0]
        else:
            pred_trigger = 0

    elif threshold_type == 'avg':
        a = pd.Series(pred)    
        win_avg = a.rolling(window=threshold_trigger).mean().to_numpy()

        c = np.where(win_avg >= threshold_prob, 1, 0)

        pred_trigger = 0
        if c.any():
            tri = np.where(c==1)
            pred_trigger = tri[0][0]-threshold_trigger+1
            pred_isTrigger = True

    elif threshold_type == 'continue':
        pred = np.where(pred >= threshold_prob, 1, 0)
        
        a = pd.Series(pred)    
        data = a.groupby(a.eq(0).cumsum()).cumsum().tolist()

        if threshold_trigger in data:
            pred_trigger = data.index(threshold_trigger)-threshold_trigger+1
            pred_isTrigger = True
        else:
            pred_trigger = 0

    left_edge = (gt_trigger - sample_tolerant) if (gt_trigger - sample_tolerant) >= 0 else 0
    right_edge = (gt_trigger + sample_tolerant) if (gt_trigger + sample_tolerant) <= 3000 else 3000
    
    # case positive 
    if (pred_trigger >= left_edge) and (pred_trigger <= right_edge) and (pred_isTrigger) and (gt_isTrigger):
        tp += 1
    elif (pred_isTrigger):
        fp += 1

    # case negative
    if (not pred_isTrigger) and (gt_isTrigger):
        fn += 1
    elif (not pred_isTrigger) and (not gt_isTrigger):
        tn += 1

    if gt_isTrigger and pred_isTrigger:
        diff.append(pred_trigger-gt_trigger)
        abs_diff.append(abs(pred_trigger-gt_trigger))

    return tp, fp, tn, fn, diff, abs_diff, pred_trigger, gt_trigger

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
    augmentations = basic_augmentations(opt, phase_dict, True)

    test_generator.add_augmentations(augmentations)

    if opt.aug:
        # data augmentation during training
        # 1) Add gaps (0.2)
        # 2) Channel dropout (0.3)
        # 3) Gaussian noise (0.5)
        # 4) Shift to end (0.3)

        gap_generator = sbg.OneOf([sbg.AddGap(), sbg.NullAugmentation()], [0.6, 0.4])
        dropout_generator = sbg.OneOf([sbg.ChannelDropout(), sbg.NullAugmentation()], [0.5, 0.5])
        noise_generator = sbg.OneOf([sbg.GaussianNoise(), sbg.NullAugmentation()], [0.5, 0.5])
        shift_generator = sbg.OneOf([sbg.ShiftToEnd(), sbg.NullAugmentation()], [0.9, 0.1])
        
        test_generator.augmentation(gap_generator)
        test_generator.augmentation(dropout_generator)
        test_generator.augmentation(noise_generator)
        test_generator.augmentation(shift_generator)

    return test_generator

if __name__ == '__main__':
    matplotlib.use('Agg')
    opt = parse_args()
    device = torch.device('cpu')

    if opt.plot:
        print(opt.save_path)
        plot_path = os.path.join('./plot', opt.save_path)
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

    # load datasets
    print('loading datasets')
    test_generator = set_generators(opt)
    test_loader = DataLoader(test_generator, batch_size=1, shuffle=False)
 
    # load model
    model = load_model(opt, device)

    output_dir = os.path.join('./results', opt.save_path)
    model_path = os.path.join(output_dir, 'model.pt')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)

    model.eval()
    # step_tp, step_fp, step_fn, step_tn = 0, 0, 0, 0
    total_plot = 0
    if opt.only_false:
        step_tp, step_tn = opt.n_plot, opt.n_plot
    elif opt.only_true:
        step_fp, step_fn = opt.n_plot, opt.n_plot
    
    if opt.n_plot == -1:
        opt.n_plot = 10000000

    idx = 0
    with tqdm(test_loader) as epoch:
        for data in epoch:
            idx += 1
            if idx <= opt.start_idx:
                continue

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

            intensity_tmp = calc_intensity(data['ori_X'][0, 0].numpy(), data['ori_X'][0, 1].numpy(), data['ori_X'][0, 2].numpy(), 'Acceleration', 100)
            
            if opt.intensity_condition:
                if snr_cur == 'noise' or intensity_tmp >= opt.intensity_upperbound or intensity_tmp <= opt.intensity_lowerbound:
                    continue

            with torch.no_grad():
                if opt.model_opt == 'basicphaseAE':
                    out = sliding_prediction(opt, model, data)[0]

                    gt = data['y'][0, 0]
                elif opt.model_opt == 'RED_PAN':
                    out_PS, out_M = model(data['X'][:, :3].to(device))
                    gt = data['X'][:, 3].squeeze().numpy()

                    out = out_PS[0, 0].detach().squeeze().cpu().numpy()                    
                else:
                    out = model(data['X'])
                    
                    if opt.model_opt == 'eqt':
                        out = out[1].squeeze().detach().numpy()
                    elif opt.model_opt == 'phaseNet':
                        out = out[0,0].detach().squeeze().cpu().numpy()
                    else:
                        out = out.squeeze().detach().numpy()
                    gt = data['y'][0, 0]

            a, b, c, d, e, f, pred_trigger, gt_trigger = evaluation(out, gt, opt.threshold_prob, opt.threshold_trigger, opt.sample_tolerant, opt.threshold_type)
            
            if a == 1:
                res = 'tp'
                # step_tp += 1
                # cur = step_tp
            elif b == 1:
                res = 'fp'
                # step_fp += 1
                # cur = step_fp
            elif c == 1:
                res = 'tn'
                # step_tn += 1
                # cur = step_tn
            else:
                res = 'fn'
                # step_fn += 1
                # cur = step_fn
            
            if opt.plot and ((res == 'fp' and not opt.only_true) or (res == 'tp' and not opt.only_false)
                or (res == 'tn' and not opt.only_false) or (res == 'fn' and not opt.only_true)) and (total_plot <= opt.n_plot):
                if opt.aug:
                    res = 'AUG_' + res

                if opt.only_true and res == 'tp':
                    plot(data["ori_X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path, ori=True)
                    plot(data["X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path)
                    
                    plot(data["X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path, zoomin=True)
                    plot(data["X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path, superzoomin=True)

                    total_plot += 1
                elif res == 'fp' or res == 'fn':
                    if (opt.only_fp and res == 'fp') or (opt.only_fn and res == 'fn') or (not opt.only_fp and not opt.only_fn):
                        plot(data["ori_X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path, ori=True)
                        plot(data["X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path)

                        plot(data["X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path, zoomin=True)
                        plot(data["X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path, superzoomin=True)

                        total_plot += 1

                elif (not opt.only_false) and (not opt.only_true):
                    plot(data["ori_X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path, ori=True)
                    plot(data["X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path)
                    
                    plot(data["X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path, zoomin=True)
                    plot(data["X"], out, gt, idx, res, pred_trigger, gt_trigger, snr_cur, intensity_tmp, plot_path, superzoomin=True)

                    total_plot += 1                    
            if total_plot > opt.n_plot:
                print('finish plotting..')
                break
        