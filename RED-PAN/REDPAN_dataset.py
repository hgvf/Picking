import numpy as np
import os
import torch
import scipy
import random
import glob
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import time
import sys
sys.path.append('/mnt/disk4/weiwei/picking_baseline/TemporalSegmentation/')
from TopDown_optimized import *

sys.path.append('/mnt/disk4/weiwei/RED-PAN/')
from gen_tar import *
import matplotlib.pyplot as plt

class REDPAN_dataset(Dataset):
    def __init__(self, basedir, option, samp_ratio, model_opt, load_to_ram=False):
        '''
        basedir: /mnt/disk4/weiwei/seismic_datasets/REDPAN_30S_pt/
        option: train or val
        samp_ratio: 要 sample 所有資料的多少百分比
        '''
        subpath = ['TW_EQ', 'TW_noise', 'STEAD']
    
        # get the file list
        seis_list = []
        if os.path.exists(os.path.join(os.path.join(basedir, subpath[0]), option)):
            if not os.path.exists(os.path.join(basedir, subpath[0]+'/'+option+'.txt')):
                seis_list = glob.glob(os.path.join(basedir, subpath[0]+'/'+option+'/*.pt'))

                print('saving seis_list: ', os.path.join(basedir, subpath[0]+'/'+option+'.txt'))
                with open(os.path.join(basedir, subpath[0]+'/'+option+'.txt'), 'w') as f:
                    for tmp in tqdm(seis_list, total=len(seis_list)):
                        f.write(tmp+'\n')
            else:
                with open(os.path.join(basedir, subpath[0]+'/'+option+'.txt'), 'r') as f:
                    seis_list = f.readlines()

        noise_list = []
        if os.path.exists(os.path.join(os.path.join(basedir, subpath[1]), option)):
            if not os.path.exists(os.path.join(basedir, subpath[1]+'/'+option+'.txt')):
                noise_list = glob.glob(os.path.join(basedir, subpath[1]+'/'+option+'/*.pt'))

                print('saving noise_list: ', os.path.join(basedir, subpath[1]+'/'+option+'.txt'))
                with open(os.path.join(basedir, subpath[1]+'/'+option+'.txt'), 'w') as f:
                    for tmp in tqdm(noise_list, total=len(noise_list)):
                        f.write(tmp+'\n')
            else:
                with open(os.path.join(basedir, subpath[1]+'/'+option+'.txt'), 'r') as f:
                    noise_list = f.readlines()

        stead_list = []
        if os.path.exists(os.path.join(os.path.join(basedir, subpath[2]), option)):
            if not os.path.exists(os.path.join(basedir, subpath[2]+'/'+option+'.txt')):
                stead_list = glob.glob(os.path.join(basedir, subpath[2]+'/'+option+'/*.pt'))

                print('saving stead_list: ', os.path.join(basedir, subpath[2]+'/'+option+'.txt'))
                with open(os.path.join(basedir, subpath[2]+'/'+option+'.txt'), 'w') as f:
                    for tmp in tqdm(stead_list, total=len(stead_list)):
                        f.write(tmp+'\n')
            else:
                with open(os.path.join(basedir, subpath[2]+'/'+option+'.txt'), 'r') as f:
                    stead_list = f.readlines()
       
        self.datalist = seis_list + noise_list + stead_list

        # preload to RAM
        self.load_to_ram = load_to_ram
        if load_to_ram:
            if option == 'val':
                option = 'dev'

            print('load to RAM...')
            basedir = os.path.join(basedir, 'preload_pt')
            files = os.listdir(basedir)

            wf, psn, mask = [], [], []
            for f in files:
                tmp = f.split('_')
                if tmp[0] == option:
                    if tmp[1] == 'wf':
                        wf.append(f)
                    elif tmp[1] == 'psn':
                        psn.append(f)
                    elif tmp[1] == 'mask': 
                        mask.append(f)

            wf.sort()
            psn.sort()
            mask.sort()

            toLoad_idx = np.arange(len(wf))
            np.random.shuffle(toLoad_idx)
            if option == 'train':
                total_len = len(toLoad_idx) * samp_ratio
                total_len = int(round(total_len))
                toLoad_idx = toLoad_idx[:total_len]

            toLoad_wf, toLoad_psn, toLoad_mask = [], [], []
            for idx in tqdm(toLoad_idx, total=len(toLoad_idx)):
                toLoad_wf.append(torch.load(os.path.join(basedir, wf[idx])))
                toLoad_psn.append(torch.load(os.path.join(basedir, psn[idx])))
                toLoad_mask.append(torch.load(os.path.join(basedir, mask[idx])))
            
            self.total_wf = torch.stack(toLoad_wf).view(-1, 3, 3000)
            self.total_psn = torch.stack(toLoad_psn).view(-1, 3, 3000)
            self.total_mask = torch.stack(toLoad_mask).view(-1, 2, 3000)
           
            del toLoad_wf, toLoad_psn, toLoad_mask

        if samp_ratio < 1:
            if not load_to_ram:
                self.idx = random.sample(range(len(self.datalist)), k=int(samp_ratio*len(self.datalist)))
            else:
                self.idx = np.arange(self.total_wf.shape[0])
        else:
            if not load_to_ram:
                self.idx = np.arange(len(self.datalist))
            else:
                self.idx = np.arange(self.total_wf.shape[0])

        self.len = len(self.idx)

        # filter-related parameters
        _filt_args = (5, [1, 45], 'bandpass', False)
        self.sos = scipy.signal.butter(*_filt_args, output="sos", fs=100.0)

        self.model_opt = model_opt

    def __getitem__(self, index):
        '''
        data preprocess: 
        1) Z-score normalization
        2) 1-45 Hz bandpass filter
        3) Characteristic, STA, LTA (if needed)
        '''

        if self.load_to_ram:
            trc_data, psn, mask = self.total_wf[self.idx[index]], self.total_psn[self.idx[index]], self.total_mask[self.idx[index]]
        else:
            # load data from disk
            data = torch.load(self.datalist[self.idx[index]].strip())
            trc_data, psn, mask = data['trc_data'], data['psn'], data['mask']
        
        # plt.subplot(311)
        # plt.plot(trc_data.T)
        # plt.subplot(312)
        # plt.plot(psn.T)
        # plt.subplot(313)
        # plt.plot(mask.T)
        # plt.savefig(f"./tmp/{time.time()}.png")
        # plt.clf()

        # zscore
        # trc_data = self._zscore(trc_data)

        # filter
        trc_data = self._filter(trc_data)
       
        # Characteristic, STA, LTA
        if self.model_opt == 'conformer' or self.model_opt == 'GRADUATE':
            trc_data = self._CharStaLta(trc_data)

        if self.model_opt == 'GRADUATE':
            # STFT
            stft = self._stft(trc_data)

            # Temporal segmentation
            seg = self._TemporalSegmentation(trc_data)

        if self.model_opt == 'GRADUATE':
            return (trc_data, psn, mask, stft, seg)
        else:
            return (trc_data, psn, mask)
            # return (torch.FloatTensor(trc_data), torch.FloatTensor(psn), torch.FloatTensor(mask))

    def __len__(self):
        return self.len

    def _zscore(self, wf):
        # demean
        wf = wf - torch.mean(wf, dim=-1, keepdims=True)

        # amp norm
        wf = wf / (torch.std(wf, dim=-1, keepdims=True) + 1e-10)

        return wf

    def _filter(self, wf):
        # default: 1-45Hz bandpass filter
        wf = scipy.signal.sosfilt(self.sos, wf, axis=-1)

        return torch.from_numpy(wf)

    def _CharStaLta(self, wf):
        CharFuncFilt = 3
        rawDataFilt = 0.939
        small_float = 1.0e-10
        STA_W = 0.6
        LTA_W = 0.015

        # filter
        result = torch.empty((wf.shape))
        data = torch.zeros(3)

        for i in range(wf.shape[1]):
            if i == 0:
                data = data * rawDataFilt + wf[:, i] + small_float
            else:
                data = (
                    data * rawDataFilt
                    + (wf[:, i] - wf[:, i - 1])
                    + small_float
                )

            result[:, i] = data

        wave_square = torch.square(result)

        # characteristic_diff
        diff = torch.empty((result.shape))

        for i in range(result.shape[1]):
            if i == 0:
                diff[:, i] = result[:, 0]
            else:
                diff[:, i] = result[:, i] - result[:, i - 1]

        diff_square = torch.square(diff)

        # characteristic's output vector
        wave_characteristic = torch.add(
            wave_square, torch.multiply(diff_square, CharFuncFilt)
        )

        # sta
        sta = torch.zeros(3)
        wave_sta = torch.empty((wf.shape))

        # Compute esta, the short-term average of edat
        for i in range(wf.shape[1]):
            sta += STA_W * (wf[:, i] - sta)

            # sta's output vector
            wave_sta[:, i] = sta

        # lta
        lta = torch.zeros(3)
        wave_lta = torch.empty((wf.shape))

        # Compute esta, the short-term average of edat
        for i in range(wf.shape[1]):
            lta += LTA_W * (wf[:, i] - lta)

            # lta's output vector
            wave_lta[:, i] = lta

        # concatenate 12-dim vector as output
        wf = torch.concat(
            (wf, wave_characteristic, wave_sta, wave_lta), axis=0
        )

        return wf

    def _stft(self, wf):
        acc = np.sqrt(wf[0]**2+wf[1]**2+wf[2]**2)
        f, t, Zxx = scipy.signal.stft(acc, nperseg=20, nfft=64)
        real = np.abs(Zxx.real).T

        return real[:, :-1]

    def _TemporalSegmentation(self, wf):
        out = TopDown(wf.clone().numpy(), 4, 1)

        if out[-1] != 4:
            out = TopDown(wf.clone().numpy(), out[-1], 1)

        seg_edge = sorted(out[0])

        gt = torch.zeros(wf.shape[-1])
        for edge in seg_edge:
            if edge == wf.shape[-1]:
                continue

            gt += gen_tar_func(wf.shape[-1], edge, 10)

        gt[gt>1] = 1

        return gt


