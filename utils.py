import seisbench.models as sbm
import seisbench.data as sbd
import seisbench.generate as sbg

import numpy as np
import sys
sys.path.append('../RED-PAN')
from gen_tar import DWA
import torch
import torch.nn.functional as F

from obspy.core.trace import Trace, Stats
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime
import obspy

from model import *
from RED_PAN_model import *

def load_dataset(opt):
    cwbsn, tsmip, stead, cwbsn_noise = 0, 0, 0, 0
    
    # loading datasets
    if opt.dataset_opt == 'stead' or opt.dataset_opt == 'all':
        # STEAD
        print('loading STEAD')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas3/earthquake_dataset_large/script/STEAD/'}}
        stead = sbd.STEAD(**kwargs)
        stead = apply_filter(stead, opt.snr_threshold, s_wave=opt.s_wave, isStead=True)

    if opt.dataset_opt == 'cwbsn' or opt.dataset_opt == 'taiwan' or opt.dataset_opt == 'all' or opt.dataset_opt == 'redpan' or opt.dataset_opt == 'prev_taiwan':
        # CWBSN 
        print('loading CWBSN')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/CWBSN/seisbench/'}}

        cwbsn = sbd.CWBSN(loading_method=opt.loading_method, **kwargs)
        cwbsn = apply_filter(cwbsn, opt.snr_threshold, True, opt.level, opt.s_wave)

    if opt.dataset_opt == 'tsmip' or opt.dataset_opt == 'taiwan' or opt.dataset_opt == 'all' or opt.dataset_opt == 'redpan' or opt.dataset_opt == 'prev_taiwan':
        # TSMIP
        print('loading TSMIP')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/TSMIP/seisbench/seisbench/'}}

        tsmip = sbd.TSMIP(loading_method=opt.loading_method, sampling_rate=100, **kwargs)

        tsmip.metadata['trace_sampling_rate_hz'] = 100
        tsmip = apply_filter(tsmip, opt.snr_threshold, s_wave=opt.s_wave)

    if opt.dataset_opt == 'stead_noise' or opt.dataset_opt == 'redpan' or opt.dataset_opt == 'prev_taiwan':
        # STEAD noise
        print('loading STEAD noise')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas3/STEAD/'}}

        stead = sbd.STEAD_noise(**kwargs)

        print('traces: ', len(stead))

    if opt.dataset_opt == 'cwbsn' or opt.dataset_opt == 'taiwan' or opt.dataset_opt == 'all':
        # CWBSN noise
        print('loading CWBSN noise')
        kwargs={'download_kwargs': {'basepath': '/mnt/disk4/weiwei/seismic_datasets/CWB_noise/'}}
        cwbsn_noise = sbd.CWBSN_noise(**kwargs)
        
        print('traces: ', len(cwbsn_noise))

    return cwbsn, tsmip, stead, cwbsn_noise

def apply_filter(data, snr_threshold=-1, isCWBSN=False, level=-1, s_wave=False, isStead=False):
    # Apply filter on seisbench.data class

    print('original traces: ', len(data))
    
    # 只選波型完整的 trace
    if not isStead:
        if isCWBSN:
            if level != -1:
                complete_mask = data.metadata['trace_completeness'] == level
            else:
                complete_mask = np.logical_or(data.metadata['trace_completeness'] == 3, data.metadata['trace_completeness'] == 4)
        else:
            complete_mask = data.metadata['trace_completeness'] == 1

        # 只選包含一個事件的 trace
        single_mask = data.metadata['trace_number_of_event'] == 1

        # making final mask
        mask = np.logical_and(single_mask, complete_mask)
        data.filter(mask)

    # 也需要 s_arrival time, 且 p, s 波距離不能太遠
    if s_wave:
        s_mask = np.logical_and(data.metadata['trace_s_arrival_sample'] <= 500000, data.metadata['trace_s_arrival_sample'] >= 0)
        data.filter(s_mask)

        ps_diff_mask = np.logical_and(data.metadata['trace_s_arrival_sample'] - data.metadata['trace_p_arrival_sample'] <= 2500, data.metadata['trace_s_arrival_sample'] - data.metadata['trace_p_arrival_sample'] > 0)
        data.filter(ps_diff_mask)

    if snr_threshold != -1:
        snr_mask = data.metadata['trace_Z_snr_db'] > snr_threshold
        data.filter(snr_mask)
        
    print('filtered traces: ', len(data))

    return data

def basic_augmentations(opt, phase_dict, test=False):
    # basic augmentations:
    #   1) Windowed around p-phase pick
    #   2) Random cut window, wavelen=3000
    #   3) Filter 
    #   4) Normalize: demean, zscore,
    #   5) Change dtype to float32
    #   6) Probabilistic: gaussian function
    
    if opt.model_opt == 'basicphaseAE':
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=2250, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Filter(N=5, Wn=[2, 25], btype='bandpass'),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0)
            ]
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=1000, windowlen=1800, selection="first", strategy="pad"),
                sbg.VtoA(),
                sbg.Filter(N=5, Wn=[2, 25], btype='bandpass'),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),            
                sbg.SlidingWindowWithLabel(timestep=40, windowlen=600),
                ]
    elif opt.model_opt == 'phaseNet':
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=2250, windowlen=3001, strategy='pad'),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std", keep_ori=True),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0)
            ]
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3001, strategy="pad"),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std"),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0)
            ]
    elif opt.model_opt == 'eqt':
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=2250, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass', keep_ori=True),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0, shape='triangle'),
            ]
            
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass'),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0, shape='triangle'),
            ]
    elif opt.model_opt == 'conformer' or opt.model_opt == 'anticopy_conformer':
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=3000-opt.p_timestep, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass', keep_ori=True),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
            
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass'),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
    elif opt.model_opt == 'conformer_embedding' or opt.model_opt == 'pretrained_embedding' or opt.model_opt == 'ssl_conformer':
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=2250, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass', keep_ori=True),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
            
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass'),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
    elif opt.model_opt == 'conformer_stft':
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=3000-opt.p_timestep, windowlen=3000, strategy='pad'),
                sbg.STFT(),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass', keep_ori=True),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
            
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                sbg.STFT(),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass'),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
    elif opt.model_opt == 'RED_PAN':
        if test:
            augmentations = [
                sbg.WindowAroundSample(['trace_p_arrival_sample'], samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=2250, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass', keep_ori=True),
                sbg.ChangeDtype(np.float32),
                sbg.RED_PAN_label(),
            ]
            
        else:
            augmentations = [
                sbg.WindowAroundSample(['trace_p_arrival_sample'], samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=2500, high=6000),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass'),
                sbg.ChangeDtype(np.float32),
                sbg.RED_PAN_label(),
            ]
    elif opt.model_opt == 'conformer_intensity':
        if test:
            augmentations = [
                sbg.Intensity(),
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=3000-opt.p_timestep, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass', keep_ori=True),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
            
        else:
            augmentations = [
                sbg.Intensity(),
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass'),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
    elif opt.model_opt == 'GRADUATE':
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=3000-opt.p_timestep, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass', keep_ori=True),
                sbg.STFT(),
                sbg.CharStaLta(),
                sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
            
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass'),
                sbg.STFT(),
                sbg.CharStaLta(),
                sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
    return augmentations

def load_model(opt, device):
    assert opt.model_opt != None, "Choose one of the model in seisbench."

    if opt.model_opt == 'eqt':
        model = sbm.EQTransformer(in_samples=3000, classes=1, phases='P')
    elif opt.model_opt == 'basicphaseAE':
        model = sbm.BasicPhaseAE(classes=2, phases='NP')
    elif opt.model_opt == 'gpd':
        model = sbm.GPD(in_channels=3, classes=1, phases='P')
    elif opt.model_opt == 'phaseNet':
        model = sbm.PhaseNet(in_channels=3, classes=2, phases='NP')
    elif opt.model_opt == 'conformer':
        rep_KV = True if opt.rep_KV == 'True' else False
        model = SingleP_Conformer(conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers, dec_layers=opt.dec_layers, 
                    d_model=opt.d_model, encoder_type=opt.encoder_type, decoder_type=opt.decoder_type, norm_type=opt.MGAN_normtype, l=opt.MGAN_l, query_type=opt.query_type,
                    rep_KV=opt.rep_KV)
    elif opt.model_opt == 'conformer_stft' or opt.model_opt == 'conformer_intensity':
        # model =  SingleP_Conformer_spectrogram(dim_spectrogram=opt.dim_spectrogram, conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers, dec_layers=opt.dec_layers, d_model=opt.d_model, encoder_type=opt.encoder_type, decoder_type=opt.decoder_type, norm_type=opt.MGAN_normtype, l=opt.MGAN_l)
        model = SingleP_Conformer(conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers, dec_layers=opt.dec_layers, 
                    d_model=opt.d_model, encoder_type=opt.encoder_type, decoder_type=opt.decoder_type, norm_type=opt.MGAN_normtype, l=opt.MGAN_l, query_type=opt.query_type, 
                    intensity_MT=opt.intensity_MT)
    elif opt.model_opt == 'transformer':
        model = SingleP_transformer_window(d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers, window_size=opt.window_size)
    elif opt.model_opt == 'conformer_embedding':
        model = SingleP_WaveformEmb_Conformer(conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers, dec_layers=opt.dec_layers, 
                    d_model=opt.d_model, encoder_type=opt.encoder_type, decoder_type=opt.decoder_type, norm_type=opt.MGAN_normtype, l=opt.MGAN_l, emb_dim=opt.emb_dim, 
                    n_class=opt.n_class)
    elif opt.model_opt == 'pretrained_embedding':
        model = SingleP_WaveformEmb(conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers, dec_layers=opt.dec_layers, d_model=opt.d_model, emb_dim=opt.emb_dim, n_class=opt.n_class,
                                    encoder_type=opt.encoder_type, decoder_type=opt.decoder_type, emb_type=opt.emb_type, pretrained_emb=opt.pretrained_emb, emb_d_ffn=opt.emb_d_ffn, emb_layers=opt.emb_layers, emb_model_opt=opt.emb_model_opt)
    elif opt.model_opt == 'ssl_conformer':
        model = SSL_Conformer(conformer_class=opt.conformer_class, d_model=opt.d_model, d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers, dec_layers=opt.dec_layers, norm_type=opt.MGAN_normtype, l=opt.MGAN_l,
                emb_dim=opt.emb_dim, emb_type=opt.emb_type, emb_d_ffn=opt.emb_d_ffn, emb_layers=opt.emb_layers, pretrained_emb=opt.pretrained_emb, decoder_type=opt.decoder_type)
    elif opt.model_opt == 'anticopy_conformer':
        model = AntiCopy_Conformer(conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers, dec_layers=opt.dec_layers, d_model=opt.d_model, encoder_type=opt.encoder_type, decoder_type=opt.decoder_type, norm_type=opt.MGAN_normtype, l=opt.MGAN_l)
    elif opt.model_opt == 'RED_PAN':
        model = RED_PAN().double()
        # model = RED_PAN()
    elif opt.model_opt == 'GRADUATE':
        rep_KV = True if opt.rep_KV == 'True' else False
        model = GRADUATE(conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, nhead=opt.nhead, d_model=opt.d_model, enc_layers=opt.enc_layers, 
                    dec_layers=opt.dec_layers, norm_type=opt.MGAN_normtype, l=opt.MGAN_l, cross_attn_type=opt.cross_attn_type, n_segmentation=opt.n_segmentation, 
                    decoder_type=opt.decoder_type, output_layer_type=opt.output_layer_type, rep_KV=rep_KV, seg_proj_type=opt.seg_proj_type)
    
    return model.to(device)

def loss_fn(opt, pred, gt, device, task_loss=None, cur_epoch=None, intensity=None):
    if opt.model_opt == 'eqt':
        weights = torch.add(torch.mul(gt[:, 0], opt.loss_weight), 1).to(device)
        
        loss = F.binary_cross_entropy(weight=weights, input=pred[1].to(device), target=gt[:, 0].type(torch.FloatTensor).to(device))
    elif opt.model_opt == 'basicphaseAE':
        # vector cross entropy loss
        h = gt.to(device) * torch.log(pred.to(device) + 1e-5)
        h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
        h = h.mean()  # Mean over batch axis

        loss = -h
    elif opt.model_opt == 'gpd':
        # TODO
        model = sbm.GPD(in_channels=3, classes=1, phases='P')
    elif opt.model_opt == 'phaseNet':
        # vector cross entropy loss
        h = gt.to(device) * torch.log(pred.to(device) + 1e-5)
        h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
        h = h.mean()  # Mean over batch axis

        loss = -h
    elif opt.model_opt == 'conformer' or opt.model_opt == 'conformer_stft' or opt.model_opt == 'conformer_embedding' or opt.model_opt == 'pretrained_embedding' \
        or opt.model_opt == 'ssl_conformer':
        weights = torch.add(torch.mul(gt[:, 0], opt.loss_weight), 1).to(device)
        
        pred = pred.squeeze()
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)

        loss = F.binary_cross_entropy(weight=weights, input=pred.to(device), target=gt[:, 0].type(torch.FloatTensor).to(device))

    elif opt.model_opt == 'conformer_intensity':
        # lambda1: picking; lambda2: multitask training
        lambda1 = 0.75
        lambda2 = 1 - lambda1

        # picking loss
        weights = torch.add(torch.mul(gt[:, 0], opt.loss_weight), 1).to(device)

        pred = pred.squeeze()
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)

        picking_loss = F.binary_cross_entropy(weight=weights, input=pred.to(device), target=gt[:, 0].type(torch.FloatTensor).to(device))

        # intensity_prediction loss
        intensity_pred, intensity_gt = intensity

        label_smoothing_factor = opt.label_smoothing

        # one-hot encoding
        target = torch.zeros(pred.shape[0], 3)
        target = target.scatter_(1, intensity_gt, 1)

        intensity_loss = F.cross_entropy(input=intensity_pred.to(device), target=target.to(device), label_smoothing=label_smoothing_factor)
        # print(f"picking_loss: {picking_loss}, intensity_loss: {intensity_loss}")
        return lambda1 * picking_loss + lambda2 * intensity_loss

    elif opt.model_opt == 'anticopy_conformer':
        import matplotlib.pyplot as plt
        import time
        def Dynamic_ignoreIndex_CELoss(pred, gt):
            loss = 0.0
            idx = (torch.arange(3000), gt.long())
            mask = torch.ones((pred.shape), dtype=torch.bool)
            mask[idx] = False
            
            ignored_pred = pred[mask].reshape(3000, 2)
            
            denom = torch.sum(torch.exp(ignored_pred), dim=-1)
            nom = torch.exp(ignored_pred[:, -1])
            loss += -torch.sum(torch.log(nom/(denom+1e-15)))/3000
            
            return loss

        loss = 0.0
        lambda1 = 0.5
        
        pred = pred.to(device)
        gt = gt[:, 0, :].to(device)
        weight = torch.ones(3).to(device)
        weight[0] = 300
        for i in range(gt.shape[0]):
            cls = torch.where(gt[i]>=0.5, True, False)
            gt[i][cls] = 0
            gt[i][~cls] = 1
            
            term1 = F.cross_entropy(pred[i], gt[i].long(), weight=weight)
            term2 = Dynamic_ignoreIndex_CELoss(pred[i], gt[i])

            loss += term1 + lambda1 * term2

        loss /= gt.shape[0]
        
        # tmp = torch.where(gt[0] == 0, True, False)
        # print('label0: ', pred[0, tmp, :])       
        # print('label1: ', pred[0, ~tmp, :10])
    elif opt.model_opt == 'RED_PAN':
        task1_loss, task2_loss = task_loss
        pred_PS, pred_M = pred
        gt_PS, gt_M = gt

        # calculate current epoch loss
        # PS, M: (batch, 3, 3000), (batch, 2, 3000)
        PS_loss, M_loss = 0.0, 0.0
        weights = torch.ones(pred_PS.shape[0], 3000).to(device)
        for i in range(3):
            # if weighted
            if i != 2:
                tmp = torch.add(torch.mul(pred_PS[:, i], opt.loss_weight), 1).to(device)
                weights += tmp
                PS_loss += F.binary_cross_entropy(weight=tmp.detach(), input=pred_PS[:, i], target=gt_PS[:, i].to(device))
            else:
                PS_loss += F.binary_cross_entropy(weight=weights.detach(), input=pred_PS[:, i], target=gt_PS[:, i].to(device))
            # PS_loss += F.binary_cross_entropy(input=pred_PS[:, i], target=gt_PS[:, i].to(device))

        weights = torch.ones(pred_M.shape[0], 3000).to(device)                
        for i in range(2):
            # if weighted
            if i != 1:
                tmp = torch.add(torch.mul(pred_M[:, i], opt.loss_weight), 1).to(device)
                weights += tmp
                M_loss += F.binary_cross_entropy(weight=tmp.detach(), input=pred_M[:, i], target=gt_M[:, i].to(device))
            else:
                M_loss += F.binary_cross_entropy(weight=weights.detach(), input=pred_M[:, i], target=gt_M[:, i].to(device))
            # M_loss += F.binary_cross_entropy(input=pred_M[:, i], target=gt_M[:, i].to(device))

        loss = DWA(task1_loss, task2_loss, cur_epoch, PS_loss, M_loss)
         
        return loss, PS_loss, M_loss
    elif opt.model_opt == 'GRADUATE':
        pred_seg, pred_picking = pred
        gt_seg, gt_picking = gt
        
        # picking loss
        weights = torch.add(torch.mul(gt_picking[:, 0], opt.loss_weight), 1).to(device)
        picking_loss = F.binary_cross_entropy(weight=weights, input=pred_picking.squeeze().to(device), target=gt_picking[:, 0].type(torch.FloatTensor).to(device))

        if opt.seg_proj_type != 'none':
            # temporal segmentation loss
            # prediction: (batch, n_segmentation, wavelength)
            # ground-truth: (batch, wavelength)
            segmentation_loss = F.cross_entropy(input=pred_seg.permute(0, 2, 1), target=gt_seg.to(device).to(torch.long), label_smoothing=opt.label_smoothing)
            print(f"picking: {picking_loss}, segmentation: {segmentation_loss}")

            loss = opt.segmentation_ratio * segmentation_loss + (1-opt.segmentation_ratio) * picking_loss
        else:
            loss = picking_loss

    return loss

def sliding_prediction(opt, model, sample):
    # input sample: (3, npts)
    predictions = []
    
    for i in range(sample['X'].shape[0]):
        data = {}
        data['X'] = sample['X'][i].numpy()

        chan = ['Z', 'N', 'E']
        for i in range(3):    
            stats = Stats()
            stats.network = 'TW'
            stats.station = 'TMP'
            stats.channel = 'HL'+chan[i]
            stats.sampling_rate = 100
            stats.starttime = UTCDateTime("2019-11-04T00:15:00.00000Z")
            stats.npts = 3000

            if i == 0:
                z = Trace(data['X'][i], header=stats)
            elif i == 1:
                n = Trace(data['X'][i], header=stats)
            else:
                e = Trace(data['X'][i], header=stats)

        argdict = model.get_model_args()
        stream = Stream(traces=[z, n, e])

        stream = stream.copy()
        stream.merge(-1)

        output = obspy.Stream()

        # # Preprocess stream, e.g., filter/resample
        model.annotate_stream_pre(stream, argdict)

        # # # Validate stream
        stream = model.annotate_stream_validate(stream, argdict)

        # # Group stream
        groups = model.group_stream(stream)

        # # Sampling rate of the data. Equal to self.sampling_rate is this is not None
        argdict["sampling_rate"] = groups[0][0].stats.sampling_rate

        stream = Stream(traces=[z, n, e])
        anno = model.annotate(stream)
        
        if opt.model_opt == 'basicphaseAE':
            pred = anno[0].data
        
        pred = np.hstack((pred, np.zeros(3000-pred.shape[0])))
        predictions.append(pred)

    return predictions
