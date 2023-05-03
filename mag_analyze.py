import json
import pickle
import bisect
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import pandas as pd
import os
from scipy.stats import gaussian_kde

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', type=str)
    parser.add_argument('--source_dir', type=str)
    parser.add_argument('--report_dir', type=str)
    parser.add_argument('--split', type=str)

    opt = parser.parse_args()

    return opt

def parse_dismag(dismag, output_dir):
    all_res = []
    df = {}
    for k, v in dismag.items():
        if k == '0':
            key = '< 50 km'
        elif k == '1':
            key = '50~80 km'
        elif k == '2':
            key = '80~100 km'
        elif k == '3':
            key = '100~150 km'
        elif k == '4':
            key = '150~200 km'
        else:
            key = '>= 200 km'
            
        df[key] = {}
        df[key]['me'] = round(np.mean(v), 4)
        df[key]['mae'] = round(np.mean(np.abs(v)), 4)
        df[key]['std'] = round(np.std(v), 4)
        df[key]['count'] = len(v)
        # print(f"{k}->{np.mean(v)}, {np.std(v)}, {len(v)}")
        
        abs_diff = np.mean(np.abs(v))
        # print(f"abs_diff: {abs_diff}")
        
        all_res += v

    df['total'] = {}
    df['total']['me'] = round(np.mean(all_res), 4)
    df['total']['mae'] = round(np.mean(np.abs(all_res)), 4)
    df['total']['std'] = round(np.std(all_res), 4)
    df['total']['count'] = len(all_res)
    # print(f"all->{np.mean(all_res)}, {np.std(all_res)}")
    abs_diff = np.mean(np.abs(all_res))
    # print(f"abs_diff: {abs_diff}")

    # save as csv
    df = pd.DataFrame.from_dict(df)
    df.to_csv(os.path.join(output_dir, 'mag_stats.csv'))

def parse_magCompare(pred, gt, output_dir):
    mag_level = [2, 3, 4, 5, 6]
    sub_keys = ['ME', 'MAE']
    stats = {}
    for i in range(len(mag_level)):
        stats[i] = {}
        
        for k in sub_keys:
            stats[i][k] = []

    pp = []
    gg = []
    for i in range(len(pred)):
        p = pred[i]
        g = gt[i]

        pp.append(p)
        gg.append(g)

        me = p - g
        mae = np.abs(me)
        
        mag_idx = bisect.bisect_right(mag_level, g)-1
        if mag_idx < 0:
            mag_idx = 0

        stats[mag_idx]['ME'].append(me)
        stats[mag_idx]['MAE'].append(mae)

    # generate result as csv
    df = {}
    for k, v in stats.items():
        if k == 0:
            key = '< 3'
        elif k == 1:
            key = '3~4'
        elif k == 2:
            key = '4~5'
        elif k == 3:
            key = '5~6'
        elif k == 4:
            key = '>= 6'

        me = round(np.mean(v['ME']), 4)
        mae = round(np.mean(v['MAE']), 4)
        std = round(np.std(v['ME']), 4)
        count = round(len(v['ME']), 4)

        df[key] = {}
        df[key]['me'] = me
        df[key]['mae'] = mae
        df[key]['std'] = std
        df[key]['count'] = count
        # print(f"{k}-> me={me}, mae={mae}, std={std}")

    # save as csv
    df = pd.DataFrame.from_dict(df)
    df.to_csv(os.path.join(output_dir, 'mag_level_compare.csv'))

    return pp, gg

def parser_magSNR(pred, gt, snr_total, output_dir):
    snr_level = [0, 10, 20, 30, 40]
    sub_keys = ['ME', 'MAE']
    stats = {}
    for i in range(len(snr_level)):
        stats[i] = {}

        for k in sub_keys:
            stats[i][k] = []
            
    for i in range(len(snr_total)):
        snr_idx = bisect.bisect_right(snr_level, snr_total[i]*10)-1
        if snr_idx < 0:
            snr_idx = 0
            
        diff = pred[i] - gt[i]
        stats[snr_idx]['ME'].append(diff)
        stats[snr_idx]['MAE'].append(np.abs(diff))
        
    df = {}
    for k, v in stats.items():
        if k == 0:
            key = '< 10 dB'
        elif k == 1:
            key = '10~20 dB'
        elif k == 2:
            key = '20~30 dB'
        elif k == 3:
            key = '30~40 dB'
        elif k == 4:
            key = '> 40 dB'
            
        df[key] = {}
        df[key]['me'] = np.mean(v['ME'])
        df[key]['mae'] = np.mean(v['MAE'])
        df[key]['std'] = np.std(v['ME'])
        df[key]['count'] = len(v['ME'])
        # print(f"{key}->{np.mean(v['ME'])}, {np.std(v['MAE'])}, {len(v['ME'])}")
                                                            
    # save as csv
    df = pd.DataFrame.from_dict(df)
    df.to_csv(os.path.join(output_dir, 'mag_SNR.csv'))

def plot_r2score(pred, gt, output_dir):
    xy = np.vstack([gt, pred])
    z = gaussian_kde(xy)(xy)

    plt.scatter(gt, pred, s=5, c=z)
    plt.xlabel('Ground-Truth Magnitude')
    plt.ylabel('Prediction Magnitude')
    plt.plot(np.arange(6), color='black')
    plt.savefig(os.path.join(output_dir, 'r2score.png'), dpi=300)
    plt.clf()

if __name__ == '__main__':
    opt = parse_args()

    output_dir = os.path.join('./results', opt.save_path)
    filedir = os.path.join(output_dir, opt.source_dir)
    stat_dir = os.path.join(output_dir, opt.report_dir)
    print('saving path: ', stat_dir)
    if not os.path.exists(stat_dir):
        os.makedirs(stat_dir)

    # load distance-magnitude json file
    if opt.split == 'test':
        filename = 'test_dismag.json'
    else:
        filename = 'dismag.json'
    
    print('opening dismag.json ...')
    f = open(os.path.join(filedir, filename), 'r')
    dismag = json.load(f)
    parse_dismag(dismag, stat_dir)

    # load prediction, ground-truth pairs through picking file
    if opt.split == 'test':
        pred_filename = 'test_mag_pred.pkl'
        gt_filename = 'test_mag_gt.pkl'
        snr_filename = 'test_snr_total.pkl'
    else:
        pred_filename = 'mag_pred.pkl'
        gt_filename = 'mag_gt.pkl'
        snr_filename = 'snr_total.pkl'

    print('opening pickle files ...')
    with open(os.path.join(filedir, pred_filename), 'rb') as f:
        pred = pickle.load(f)
    with open(os.path.join(filedir, gt_filename), 'rb') as f:
        gt = pickle.load(f)
    with open(os.path.join(filedir, snr_filename), 'rb') as f:
        snr_total = pickle.load(f)
        
    parser_magSNR(pred, gt, snr_total, stat_dir)
    pred, gt = parse_magCompare(pred, gt, stat_dir)
    plot_r2score(pred, gt, stat_dir)

    
