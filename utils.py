import seisbench.models as sbm
import seisbench.data as sbd
import seisbench.generate as sbg

import numpy as np
import sys
sys.path.append('./RED-PAN')
from gen_tar import DWA
import torch
import torch.nn.functional as F

from obspy.core.trace import Trace, Stats
from obspy.core.stream import Stream
from obspy.core.utcdatetime import UTCDateTime
import obspy

from model import *
from RED_PAN_model import *
from tsfc_modules import TSFC_Unet
from wmseg_dataparallel import BalancedDataParallel

def load_dataset(opt):
    cwbsn, tsmip, stead, cwbsn_noise, instance = 0, 0, 0, 0, 0
    
    if opt.dataset_opt == 'instance' or opt.dataset_opt == 'all':
        print('loading INSTANCE')
        kwargs={'download_kwargs': {'basepath': '/home/weiwei/disk4/seismic_datasets/'}}
        instance = sbd.InstanceCountsCombined(**kwargs)

        instance = apply_filter(instance, isINSTANCE=True, filter_instance=opt.filter_instance)

    # loading datasets
    if opt.dataset_opt == 'stead' or opt.dataset_opt == 'all':
        # STEAD
        print('loading STEAD')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas3/earthquake_dataset_large/script/STEAD/'}}
        stead = sbd.STEAD(**kwargs)

        # stead = apply_filter(stead, snr_threshold=opt.snr_threshold, s_wave=opt.s_wave, isStead=True, magnitude=True)
        stead = apply_filter(stead, snr_threshold=opt.snr_threshold, s_wave=opt.s_wave, isStead=True)

    if opt.dataset_opt == 'cwbsn' or opt.dataset_opt == 'taiwan' or opt.dataset_opt == 'all' or opt.dataset_opt == 'redpan' or opt.dataset_opt == 'prev_taiwan' or opt.dataset_opt == 'EEW':
        # CWBSN 
        print('loading CWBSN')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/CWBSN/seisbench/'}}

        cwbsn = sbd.CWBSN(loading_method=opt.loading_method, **kwargs)
        cwbsn = apply_filter(cwbsn, snr_threshold=opt.snr_threshold, isCWBSN=True, level=opt.level, s_wave=opt.s_wave, instrument=opt.instrument, location=opt.location)

    if opt.dataset_opt == 'tsmip' or opt.dataset_opt == 'taiwan' or opt.dataset_opt == 'all' or opt.dataset_opt == 'redpan' or opt.dataset_opt == 'prev_taiwan' or opt.dataset_opt == 'EEW':
        # TSMIP
        print('loading TSMIP') 
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/TSMIP/seisbench/seisbench/'}}

        tsmip = sbd.TSMIP(loading_method=opt.loading_method, sampling_rate=100, **kwargs)

        tsmip.metadata['trace_sampling_rate_hz'] = 100
        tsmip = apply_filter(tsmip, snr_threshold=opt.snr_threshold, s_wave=opt.s_wave, instrument=opt.instrument, location=opt.location)

    if opt.dataset_opt == 'stead_noise' or opt.dataset_opt == 'redpan' or opt.dataset_opt == 'prev_taiwan' or opt.dataset_opt == 'cwbsn':
        # STEAD noise
        print('loading STEAD noise')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas3/STEAD/'}}

        stead = sbd.STEAD_noise(**kwargs)

        print('traces: ', len(stead))

    if opt.dataset_opt == 'cwbsn' or opt.dataset_opt == 'taiwan' or opt.dataset_opt == 'all' or opt.dataset_opt == 'EEW' or opt.dataset_opt == 'cwb_noise':
        # CWBSN noise
        print('loading CWBSN noise')
        kwargs={'download_kwargs': {'basepath': '/mnt/disk4/weiwei/seismic_datasets/CWB_noise/'}}
        cwbsn_noise = sbd.CWBSN_noise(**kwargs)
        
        cwbsn_noise = apply_filter(cwbsn_noise, instrument=opt.instrument, isNoise=True, noise_sample=opt.noise_sample, location=opt.location)

        if opt.dataset_opt == 'EEW':
            cwbsn_noise = apply_filter(cwbsn_noise, instrument='HL', isNoise=True)

        print('traces: ', len(cwbsn_noise))

    return cwbsn, tsmip, stead, cwbsn_noise, instance

def apply_filter(data, snr_threshold=-1, isCWBSN=False, level=-1, s_wave=False, isStead=False, isNoise=False, instrument='all', noise_sample=200000, magnitude=False, location=-1, isINSTANCE=False, filter_instance=False):
    # Apply filter on seisbench.data class

    print('original traces: ', len(data))
    
    # 只選波型完整的 trace
    if not isStead and not isNoise and not isINSTANCE:
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

    if location != -1:
        location_mask = np.logical_or(data.metadata['station_location_code'] == location, data.metadata['station_location_code'] == str(location))
        data.filter(location_mask)

    # 篩選儀器
    if instrument != 'all':
        instrument_mask = data.metadata["trace_channel"] == instrument
        data.filter(instrument_mask)

    if isNoise and noise_sample != -1:
        noise_mask = data.metadata['Unnamed: 0'] < noise_sample
        data.filter(noise_mask)

    if isStead and magnitude:
        magnitude_earthquake_mask = np.logical_and(data.metadata['source_magnitude_type'] == 'ml', data.metadata['trace_category'] == 'earthquake_local')
        magnitude_noise_mask = data.metadata['trace_category'] == 'noise'
        magnitude_mask = np.logical_or(magnitude_earthquake_mask, magnitude_noise_mask)
        data.filter(magnitude_mask)

    if isINSTANCE and filter_instance:
        p_weight_mask = data.metadata['path_weight_phase_location_P'] >= 50
        eqt_mask = np.logical_and(data.metadata['trace_EQT_number_detections'] == 1, data.metadata['trace_EQT_P_number'] == 1)
        instance_mask = np.logical_and(p_weight_mask, eqt_mask)
        data.filter(instance_mask)

    print('filtered traces: ', len(data))

    return data

def basic_augmentations(opt, phase_dict, ptime=None, test=False, EEW=False):
    # basic augmentations:
    #   1) Windowed around p-phase pick
    #   2) Random cut window, wavelen=3000
    #   3) Filter 
    #   4) Normalize: demean, zscore,
    #   5) Change dtype to float32
    #   6) Probabilistic: gaussian function
    
    if opt.seg_proj_type == 'none':
        seg_null = True
    else:
        seg_null = False

    if opt.dataset_opt == 'instance':
        p_phases = 'trace_P_arrival_sample'
        s_phases = 'trace_S_arrival_sample'
    else:
        p_phases = 'trace_p_arrival_sample'
        s_phases = 'trace_s_arrival_sample'

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
        phase_dict = [p_phases, s_phases]
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=3001-ptime, windowlen=3001, strategy='pad'),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std", keep_ori=True),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0)
            ]
        elif EEW:
            augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.RandomWindow(windowlen=3001, strategy="pad", low=100, high=3300),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3001, strategy="pad"),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std"),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0)
            ]
    elif opt.model_opt == 'eqt':
        phase_dict = [p_phases, s_phases]

        if test:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    # sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.FixedWindow(p0=opt.wavelength-ptime, windowlen=opt.wavelength, strategy='pad'),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=opt.wavelength-ptime, windowlen=opt.wavelength, strategy='pad'),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
        elif EEW:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=330),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
            else:
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=3300),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
        else:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    # sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=opt.wavelength*0.1, high=opt.wavelength*2),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
            else:
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    # sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=opt.wavelength*0.1, high=opt.wavelength*2),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
    elif opt.model_opt == 'tsfc':
        p_phases = 'trace_p_arrival_sample'
        s_phases = 'trace_s_arrival_sample'
        phase_dict = [p_phases, s_phases]

        if test:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                    #sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.STFT(),
                    sbg.Magnitude(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
            else:
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                    sbg.VtoA(),
                    #sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                    sbg.STFT(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.Magnitude(),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
        else:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                    #sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.STFT(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.Magnitude(),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
            else:
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                    sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.STFT(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.Magnitude(),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
    elif opt.model_opt == 'conformer' or opt.model_opt == 'anticopy_conformer':
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
        elif EEW:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=4000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=3500),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),]
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
    elif opt.model_opt == 'conformer_embedding' or opt.model_opt == 'pretrained_embedding' or opt.model_opt == 'ssl_conformer':
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
            
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
    elif opt.model_opt == 'conformer_stft':
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                sbg.STFT(),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
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
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
    elif opt.model_opt == 'RED_PAN':
        if test:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.ChangeDtype(np.float32),
                    sbg.RED_PAN_label(p_label_columns=p_phases, s_label_columns=s_phases),
                ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                    sbg.ChangeDtype(np.float32),
                    sbg.RED_PAN_label(p_label_columns=p_phases, s_label_columns=s_phases),
                ]
        elif EEW:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=330),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.ChangeDtype(np.float32),
                    sbg.RED_PAN_label(p_label_columns=p_phases, s_label_columns=s_phases),
                ]
            else:
                augmentations = [
                    sbg.OneOf([sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"), sbg.NullAugmentation()],probabilities=[2, 1]),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=3300),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.ChangeDtype(np.float32),
                    sbg.RED_PAN_label(p_label_columns=p_phases, s_label_columns=s_phases),
                ]
        else:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.ChangeDtype(np.float32),
                    sbg.RED_PAN_label(p_label_columns=p_phases, s_label_columns=s_phases),
                ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.ChangeDtype(np.float32),
                    sbg.RED_PAN_label(p_label_columns=p_phases, s_label_columns=s_phases),
                ]
    elif opt.model_opt == 'conformer_intensity':
        if test:
            augmentations = [
                sbg.Intensity(),
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
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
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
    elif opt.model_opt == 'conformer_noNorm':
        if test:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
        elif EEW:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=4000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=3300),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),]
        else:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                sbg.CharStaLta(),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
            ]
    elif opt.model_opt == 'GRADUATE' or opt.model_opt == 'ensemble_picker':
        if opt.label_type == 'all':
            phase_dict = [p_phases, s_phases]
        else:
            phase_dict = [p_phases]

        if test:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=opt.wavelength-ptime, windowlen=opt.wavelength, strategy='pad'),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.STFT(max_freq=opt.max_freq),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=opt.wavelength-ptime, windowlen=opt.wavelength, strategy='pad'),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                    sbg.STFT(max_freq=opt.max_freq),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                ]
        elif EEW:
            if opt.wavelength == 3000:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=3300),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.STFT(max_freq=opt.max_freq),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=100, high=opt.wavelength+300),
                    # sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.STFT(max_freq=opt.max_freq),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),]
        else:
            if opt.dataset_opt == 'stead':
                augmentations = [
                        sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                        sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=opt.wavelength*0.1, high=opt.wavelength*2),
                        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                        sbg.STFT(max_freq=opt.max_freq),
                        sbg.CharStaLta(),
                        sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
                        sbg.ChangeDtype(np.float32),
                        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                ]
            else:
                augmentations = [
                        sbg.WindowAroundSample(phase_dict, samples_before=opt.wavelength, windowlen=opt.wavelength*2, selection="first", strategy="pad"),
                        sbg.RandomWindow(windowlen=opt.wavelength, strategy="pad", low=opt.wavelength*0.1, high=opt.wavelength*2),
                        # sbg.VtoA(),
                        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                        sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                        sbg.STFT(max_freq=opt.max_freq),
                        sbg.CharStaLta(),
                        sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation, null=seg_null),
                        sbg.ChangeDtype(np.float32),
                        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                ]
        
        if opt.label_type == 'all':
            augmentations.append(sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4))
    elif opt.model_opt == 'GRADUATE_MAG' or opt.model_opt == 'GRADUATE_MAG_noNorm' or opt.model_opt == 'GRADUATE_MAG24' or opt.model_opt == 'GRADUATE_MAG_deStationary':
        p_phases = 'trace_p_arrival_sample'
        s_phases = 'trace_s_arrival_sample'
        
        if opt.label_type == 'all':
            phase_dict = [p_phases, s_phases]
        else:
            phase_dict = [p_phases]

        if test:
            augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
            ]
            if opt.model_opt == 'GRADUATE_MAG':
                augmentations.append(sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),)
            elif opt.model_opt == 'GRADUATE_MAG_deStationary':
                augmentations.append(sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_mean_std=True))
            else:
                augmentations.append(sbg.Normalize(demean_axis=-1, keep_ori=True),)

            if not opt.dataset_opt == 'stead':
                augmentations.append(sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),)
            augmentations += [sbg.STFT(),
                              sbg.FFT(),
                              sbg.Magnitude(),
                            ]
            if opt.model_opt == 'GRADUATE_MAG24':
                augmentations.append(sbg.CharStaLta(train=True))
            else:
                augmentations.append(sbg.CharStaLta())

            augmentations += [sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                              sbg.ChangeDtype(np.float32),
                              sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                             ]
        elif EEW:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=3300),
                sbg.VtoA(),
            ]
            if opt.model_opt == 'GRADUATE_MAG':
                augmentations.append(sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),)
            elif opt.model_opt == 'GRADUATE_MAG24':
                augmentations.append(sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),)
            elif opt.model_opt == 'GRADUATE_MAG_deStationary':
                augmentations.append(sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_mean_std=True))
            else:
                augmentations.append(sbg.Normalize(demean_axis=-1),)

            augmentation += [
                                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                                sbg.STFT(),
                                sbg.FFT(),
                                sbg.Magnitude(),
                                sbg.CharStaLta(),
                                sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                                sbg.ChangeDtype(np.float32),
                                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                            ]
        else:
            augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
            ]
            if opt.model_opt == 'GRADUATE_MAG':
                augmentations.append(sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),)
            elif opt.model_opt == 'GRADUATE_MAG24':
                augmentations.append(sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),)
            elif opt.model_opt == 'GRADUATE_MAG_deStationary':
                augmentations.append(sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_mean_std=True))
            else:
                augmentations.append(sbg.Normalize(demean_axis=-1),)

            if not opt.dataset_opt == 'stead':
                augmentations.append(sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),)
            augmentations += [sbg.STFT(),
                              sbg.FFT(),
                              sbg.Magnitude(),
                            ]
            if opt.model_opt == 'GRADUATE_MAG24':
                augmentations.append(sbg.CharStaLta(train=True))
            else:
                augmentations.append(sbg.CharStaLta())

            augmentations += [sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                              sbg.ChangeDtype(np.float32),
                              sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),
                             ]
        
        if opt.label_type == 'all':
            augmentations.append(sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4))
    elif opt.model_opt == 'real_GRADUATE':
        p_phases = 'trace_p_arrival_sample'
        s_phases = 'trace_s_arrival_sample'
        
        phase_dict = [p_phases, s_phases]

        if test:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_mean_std=True, keep_ori=True,),
                    sbg.STFT(),
                    sbg.FFT(),
                    sbg.Magnitude(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                    sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_mean_std=True, keep_ori=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                    sbg.STFT(),
                    sbg.FFT(),
                    sbg.Magnitude(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
        elif EEW:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=4000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=3300),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_mean_std=True),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                sbg.STFT(),
                sbg.FFT(),
                sbg.Magnitude(),
                sbg.CharStaLta(),
                sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),]
        else:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_mean_std=True),
                    sbg.STFT(),
                    sbg.FFT(),
                    sbg.Magnitude(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
            ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                    sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_type='std', amp_norm_axis=-1, keep_mean_std=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.STFT(),
                    sbg.FFT(),
                    sbg.Magnitude(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
    elif opt.model_opt == 'real_GRADUATE_noNorm':
        p_phases = 'trace_p_arrival_sample'
        s_phases = 'trace_s_arrival_sample'
        
        phase_dict = [p_phases, s_phases]

        if test:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                    sbg.Normalize(demean_axis=-1, keep_ori=True),
                    sbg.STFT(),
                    sbg.FFT(),
                    sbg.Magnitude(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                    sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, keep_ori=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                    sbg.STFT(),
                    sbg.FFT(),
                    sbg.Magnitude(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
        elif EEW:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=4000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=3300),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                sbg.STFT(),
                sbg.FFT(),
                sbg.CharStaLta(),
                sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),]
        else:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                    sbg.Normalize(demean_axis=-1),
                    sbg.STFT(),
                    sbg.FFT(),
                    sbg.Magnitude(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
            ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                    sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.STFT(),
                    sbg.FFT(),
                    sbg.Magnitude(),
                    sbg.CharStaLta(),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
    elif opt.model_opt == 'real_GRADUATE_noNorm_double':
        p_phases = 'trace_p_arrival_sample'
        s_phases = 'trace_s_arrival_sample'
        
        phase_dict = [p_phases, s_phases]

        if test:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.STFT(key="ori_X"),
                    sbg.FFT(key="ori_X"),
                    sbg.Magnitude(),
                    sbg.CharStaLta(train=True),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.FixedWindow(p0=3000-ptime, windowlen=3000, strategy='pad'),
                    sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass', keep_ori=True),
                    sbg.STFT(key="ori_X"),
                    sbg.FFT(key="ori_X"),
                    sbg.Magnitude(),
                    sbg.CharStaLta(train=True),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
        elif EEW:
            augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=4000, selection="first", strategy="pad"),
                sbg.RandomWindow(windowlen=3000, strategy="pad", low=100, high=3300),
                sbg.VtoA(),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                sbg.STFT(key="ori_X"),
                sbg.FFT(key="ori_X"),
                sbg.CharStaLta(train=True),
                sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                sbg.ChangeDtype(np.float32),
                sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=10, dim=0),]
        else:
            if opt.dataset_opt == 'stead':
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.STFT(key="ori_X"),
                    sbg.FFT(key="ori_X"),
                    sbg.Magnitude(),
                    sbg.CharStaLta(train=True),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
            ]
            else:
                augmentations = [
                    sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                    sbg.RandomWindow(windowlen=3000, strategy="pad", low=950, high=6000),
                    sbg.VtoA(),
                    sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', keep_ori=True),
                    sbg.Filter(N=5, Wn=[1, 10], btype='bandpass'),
                    sbg.STFT(key="ori_X"),
                    sbg.FFT(key="ori_X"),
                    sbg.Magnitude(),
                    sbg.CharStaLta(train=True),
                    sbg.TemporalSegmentation(n_segmentation=opt.n_segmentation),
                    sbg.ChangeDtype(np.float32),
                    sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=20, dim=0),
                    sbg.DetectionLabeller(p_phases, s_phases, key=("X", "detections"), factor=1.4)
                ]
    return augmentations

def load_model(opt, device):
    assert opt.model_opt != None, "Choose one of the model in seisbench."

    if opt.model_opt == 'real_GRADUATE':
        model = REAL_GRADUATE(d_model=opt.d_model, conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers)
    elif opt.model_opt == 'real_GRADUATE_noNorm':
        model = REAL_GRADUATE_noNorm(d_model=opt.d_model, conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers)
    elif opt.model_opt == 'real_GRADUATE_noNorm_double':
        model = REAL_GRADUATE_doubleWave(d_model=opt.d_model, conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers)
    elif opt.model_opt == 'GRADUATE_MAG' or opt.model_opt == 'GRADUATE_MAG_noNorm' or opt.model_opt == 'GRADUATE_MAG24':
        rep_KV = True if opt.rep_KV == 'True' else False
        model = GRADUATE_MAG(conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, nhead=opt.nhead, d_model=opt.d_model, enc_layers=opt.enc_layers, 
                    encoder_type=opt.encoder_type, dec_layers=opt.dec_layers, cross_attn_type=opt.cross_attn_type, 
                    decoder_type=opt.decoder_type, rep_KV=rep_KV, seg_proj_type=opt.seg_proj_type,
                    label_type=opt.label_type, res_dec=opt.res_dec)
    elif opt.model_opt == 'GRADUATE_MAG_deStationary':
        rep_KV = True if opt.rep_KV == 'True' else False
        model = GRADUATE_MAG_deStationary(conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, nhead=opt.nhead, d_model=opt.d_model, enc_layers=opt.enc_layers, 
                    encoder_type=opt.encoder_type, dec_layers=opt.dec_layers, cross_attn_type=opt.cross_attn_type, 
                    decoder_type=opt.decoder_type, rep_KV=rep_KV, seg_proj_type=opt.seg_proj_type,
                    label_type=opt.label_type, res_dec=opt.res_dec)
    elif opt.model_opt == 'tsfc':
        model = TSFC_Unet(isConformer=opt.isConformer)
    elif opt.model_opt == 'eqt':
        model = sbm.EQTransformer(in_samples=opt.wavelength, isConformer=opt.isConformer, conformer_class=opt.conformer_class, conformer_d_ffn=opt.d_ffn, 
                        conformer_nhead=opt.nhead, conformer_layers=opt.enc_layers)
    elif opt.model_opt == 'basicphaseAE':
        model = sbm.BasicPhaseAE(classes=2, phases='NP')
    elif opt.model_opt == 'gpd':
        model = sbm.GPD(in_channels=3, classes=1, phases='P')
    elif opt.model_opt == 'phaseNet':
        model = sbm.PhaseNet(in_channels=3, classes=3, phases='NPS')
    elif opt.model_opt == 'conformer' or opt.model_opt == 'conformer_noNorm':
        rep_KV = True if opt.rep_KV == 'True' else False
        model = SingleP_Conformer(conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, n_head=opt.nhead, enc_layers=opt.enc_layers, dec_layers=opt.dec_layers, 
                    d_model=opt.d_model, encoder_type=opt.encoder_type, decoder_type=opt.decoder_type, norm_type=opt.MGAN_normtype, l=opt.MGAN_l, query_type=opt.query_type,
                    rep_KV=opt.rep_KV, label_type=opt.label_type)
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
        # model = mtan_R2unet().double()
    elif opt.model_opt == 'GRADUATE':
        rep_KV = True if opt.rep_KV == 'True' else False
        model = GRADUATE(conformer_class=opt.conformer_class, d_ffn=opt.d_ffn, nhead=opt.nhead, d_model=opt.d_model, enc_layers=opt.enc_layers, 
                    encoder_type=opt.encoder_type, dec_layers=opt.dec_layers, norm_type=opt.MGAN_normtype, l=opt.MGAN_l, cross_attn_type=opt.cross_attn_type, 
                    decoder_type=opt.decoder_type, rep_KV=rep_KV, seg_proj_type=opt.seg_proj_type,
                    label_type=opt.label_type, recover_type=opt.recover_type, rep_query=opt.rep_query, input_type=opt.input_type, 
                    stft_loss=opt.stft_loss, patch_crossattn=opt.patch_crossattn, max_freq=opt.max_freq, wavelength=opt.wavelength, stft_recovertype=opt.stft_recovertype,
                    stft_residual=opt.stft_residual, dualDomain_type=opt.dualDomain_type, ablation=opt.ablation)
    elif opt.model_opt == 'ensemble_picker':
        model = Ensemble_picker(ensemble_opt=opt.ensemble_opt, freeze_picker=opt.freeze_picker, eqt_path=opt.eqt_path, graduate_path=opt.graduate_path, redpan_path=opt.redpan_path)

    return model.to(device)
    # return BalancedDataParallel(20, model.to(device))

def loss_fn(opt, pred, gt, device, task_loss=None, cur_epoch=None, intensity=None, eqt_regularization=None, stft_intermediate=None):
    
    if opt.model_opt == 'tsfc':
        pred_seg, pred_mag, pred_picking = pred
        seg_gt, mag_gt, picking_gt, detection_gt = gt
        loss_weight = [0.6, 0.35, 0.05]
        reduction = 'mean'

        # ground-truth -> 0: P-phase, 1: S-phase, 2: other label
        # detection
        # prediction -> 0: detection, 1: P-phase picking, 2: S-phase picking
        picking_loss = 0.0
        for i in range(3):
            if i == 0 or i == 1:
                nonzero_idx = (picking_gt[:, i] != 0)
                weights = torch.ones(picking_gt[:, i].shape)*0.11
                weights[nonzero_idx] = 0.89 if opt.loss_weight == -1 else opt.loss_weight 

                picking_loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred_picking[i+1].to(device), target=picking_gt[:, i].type(torch.FloatTensor).to(device), reduction=reduction)
            else:
                nonzero_idx = (detection_gt[:, 0] != 0)
                weights = torch.ones(detection_gt[:, 0].shape)*0.11
                weights[nonzero_idx] = 0.89 if opt.loss_weight == -1 else opt.loss_weight 

                picking_loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred_picking[0].to(device), target=detection_gt[:, 0].type(torch.FloatTensor).to(device), reduction=reduction)

        # temporal segmentation
        nonzero_idx = (seg_gt != 0)
        weights = torch.ones(seg_gt.shape)
        weights[nonzero_idx] = 10

        segmentation_loss = F.binary_cross_entropy(input=pred_seg[:, 0], target=seg_gt.type(torch.FloatTensor).to(device), weight=weights.to(device), reduction=reduction)
            
        # magnitude estimation
        weights = torch.max(mag_gt, dim=1).values
        # weights[weights==0] = -2
        # mag_gt[mag_gt==0] = -2

        if pred_mag.ndim == 3:
            # sequence regression
            tmp_loss = F.l1_loss(input=pred_mag[:, :, 0], target=mag_gt.type(torch.FloatTensor).to(device), reduction='none')
            tmp_loss = torch.mean(tmp_loss, dim=-1)
        else:
            # single regression
            mag_gt = weights.clone()
            weights = weights * 2.5
            weights[weights==0] = 1

            tmp_loss = F.l1_loss(input=pred_mag, target=mag_gt.type(torch.FloatTensor).to(device), reduction='none')

        magnitude_loss = (tmp_loss * weights.to(device)).mean()

        loss = opt.segmentation_ratio * segmentation_loss + opt.magnitude_ratio * magnitude_loss + (1-opt.segmentation_ratio-opt.magnitude_ratio) * picking_loss
        # print(f"mag_gt: {mag_gt}, mag_pred: {pred_mag}")
        # print(f"segmentation: {segmentation_loss}, 'magnitude: {magnitude_loss}, picking: {picking_loss}")

        reg = 0.0
        l1 = 1e-4
        model, eqt_reg = eqt_regularization
        for name, param in model.state_dict().items():
            if name+'\n' in eqt_reg:
                reg += torch.norm(param.data, 1)
        
        loss = loss + reg * l1
    elif opt.model_opt == 'eqt':
        picking_gt, detection_gt = gt
        loss_weight = [0.4, 0.55, 0.05]       # P, S, detection
        reduction = 'sum'

        # ground-truth -> 0: P-phase, 1: S-phase, 2: other label
        # detection
        # prediction -> 0: detection, 1: P-phase picking, 2: S-phase picking
        loss = 0.0
        for i in range(3):
            if i == 0 or i == 1:
                nonzero_idx = (picking_gt[:, i] != 0)
                weights = torch.ones(picking_gt[:, i].shape)*0.11
                weights[nonzero_idx] = 0.89

                loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred[i+1].to(device), target=picking_gt[:, i].type(torch.FloatTensor).to(device), reduction=reduction)
            else:
                nonzero_idx = (detection_gt[:, 0] != 0)
                weights = torch.ones(detection_gt[:, 0].shape)*0.11
                weights[nonzero_idx] = 0.89

                loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred[0].to(device), target=detection_gt[:, 0].type(torch.FloatTensor).to(device), reduction=reduction)

        reg = 0.0
        l1 = 1e-4
        model, eqt_reg = eqt_regularization
        for name, param in model.state_dict().items():
            if name+'\n' in eqt_reg:
                reg += torch.norm(param.data, 1)
        
        loss = loss + reg * l1
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
        or opt.model_opt == 'ssl_conformer' or opt.model_opt == 'conformer_noNorm':
        
        if opt.label_type == 'p':
            weights = torch.add(torch.mul(gt[:, 0], opt.loss_weight), 1).to(device)
            loss = F.binary_cross_entropy(weight=weights, input=pred[:, :, 0].to(device), target=gt[:, 0].type(torch.FloatTensor).to(device))
        elif opt.label_type == 'other':
            loss = 0.0
            weights = torch.add(torch.mul(gt[:, 0], opt.loss_weight), 1).to(device)
            for i in range(2):
                # 0: picking label, 1: noise label
                loss += F.binary_cross_entropy(weight=weights, input=pred[:, :, i].to(device), target=gt[:, i].type(torch.FloatTensor).to(device))
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
        # PS_loss, M_loss = 0.0, 0.0
        # weights = torch.ones(pred_PS.shape[0], 3000).to(device)
        # for i in range(3):
        #     # if weighted
        #     if i != 2:
        #         tmp = torch.add(torch.mul(pred_PS[:, i], opt.loss_weight), 1).to(device)
        #         weights += tmp
        #         PS_loss += F.binary_cross_entropy(weight=tmp.detach(), input=pred_PS[:, i], target=gt_PS[:, i].type(torch.DoubleTensor).to(device))
        #     else:
        #         PS_loss += F.binary_cross_entropy(weight=weights.detach(), input=pred_PS[:, i], target=gt_PS[:, i].type(torch.DoubleTensor).to(device))
            # PS_loss += F.binary_cross_entropy(input=pred_PS[:, i], target=gt_PS[:, i].to(device))

        PS_loss = (-pred_PS.log() * gt_PS.to(device)).mean(dim=(1, 2)).mean()

        # weights = torch.ones(pred_M.shape[0], 3000).to(device)                
        # for i in range(2):
        #     # if weighted
        #     if i != 1:
        #         tmp = torch.add(torch.mul(pred_M[:, i], opt.loss_weight), 1).to(device)
        #         weights += tmp
        #         M_loss += F.binary_cross_entropy(weight=tmp.detach(), input=pred_M[:, i], target=gt_M[:, i].type(torch.DoubleTensor).to(device))
        #     else:
        #         M_loss += F.binary_cross_entropy(weight=weights.detach(), input=pred_M[:, i], target=gt_M[:, i].type(torch.DoubleTensor).to(device))
            # M_loss += F.binary_cross_entropy(input=pred_M[:, i], target=gt_M[:, i].to(device))

        M_loss = (-pred_M.log() * gt_M.to(device)).mean(dim=(1, 2)).mean()

        loss = DWA(task1_loss, task2_loss, cur_epoch, PS_loss, M_loss)

        return loss, PS_loss, M_loss
    elif opt.model_opt == 'GRADUATE':
        pred_seg, pred_picking = pred
        reduction = 'mean'

        # ======================== Picking ======================= #
        if opt.label_type == 'p':
            gt_seg, gt_picking = gt

            weights = torch.add(torch.mul(gt_picking[:, 0], opt.loss_weight), 1).to(device)
            picking_loss = F.binary_cross_entropy(weight=weights, input=pred_picking[:, :, 0].to(device), target=gt_picking[:, 0].type(torch.FloatTensor).to(device), reduction=reduction)
        elif opt.label_type == 'other':
            gt_seg, gt_picking = gt

            picking_loss = 0.0
            weights = torch.add(torch.mul(gt_picking[:, 0], opt.loss_weight), 1).to(device)
            for i in range(2):
                picking_loss += F.binary_cross_entropy(weight=weights, input=pred_picking[:, :, i].squeeze().to(device), target=gt_picking[:, i].type(torch.FloatTensor).to(device), reduction=reduction)
        elif opt.label_type == 'all':
            pred_detection, pred_p, pred_s = pred_picking
            
            pred = [pred_p, pred_s]
            gt_seg, gt_picking, gt_detection = gt
            
            loss_weight = [0.6, 0.35, 0.05]       # P, S, detection
        
            # ground-truth -> 0: P-phase, 1: S-phase, 2: other label
            # detection
            # prediction -> 0: detection, 1: P-phase picking, 2: S-phase picking
            picking_loss = 0.0
            for i in range(3):
                if i == 0 or i == 1:
                    weights = torch.add(torch.mul(gt_picking[:, i], opt.loss_weight), 1).to(device)
                    picking_loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred[i].squeeze().to(device), target=gt_picking[:, i].type(torch.FloatTensor).to(device), reduction=reduction)
                else:
                    weights = torch.add(torch.mul(gt_detection[:, 0], opt.loss_weight), 1).to(device)
                    picking_loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred_detection.squeeze().to(device), target=gt_detection[:, 0].type(torch.FloatTensor).to(device), reduction=reduction)

            if opt.stft_loss:
                for i in range(2):
                    weights = torch.add(torch.mul(gt_picking[:, i], opt.loss_weight), 1).to(device)
                    picking_loss += 0.5 * F.binary_cross_entropy(weight=weights.to(device), input=stft_intermediate[:, :, i].squeeze().to(device), target=gt_picking[:, i].type(torch.FloatTensor).to(device), reduction=reduction)

        # ====================== Temporal segmentation ======================== #
        if opt.seg_proj_type != 'none':
            # temporal segmentation loss
            # prediction: (batch, wavelength, 1)
            # ground-truth: (batch, wavelength)

            loss_weights = [0.65, 0.35]
            weights = torch.add(torch.mul(gt_seg, opt.loss_weight), 1).to(device)
            seg_bce_loss = F.binary_cross_entropy(input=pred_seg[:, :, 0], target=gt_seg.type(torch.FloatTensor).to(device), weight=weights, reduction=reduction)
            
            # n_seg = torch.FloatTensor([l.count(1) for l in gt_seg.tolist()]).unsqueeze(-1)
            # n_seg = n_seg.repeat(1, 3000).to(device)
            # seg_quantized_loss = F.l1_loss(input=pred_seg[:, :, 0], target=n_seg)

            # segmentation_loss = loss_weights[0] * seg_bce_loss + loss_weights[1] * seg_quantized_loss
            segmentation_loss = seg_bce_loss
            # print(f"picking: {picking_loss}, segmentation: {segmentation_loss}")

            loss = opt.segmentation_ratio * segmentation_loss + (1-opt.segmentation_ratio) * picking_loss
        else:
            loss = picking_loss
    elif opt.model_opt == 'GRADUATE_MAG' or opt.model_opt == 'GRADUATE_MAG_noNorm' or opt.model_opt == 'GRADUATE_MAG24' or opt.model_opt == 'GRADUATE_MAG_deStationary':
        pred_seg, pred_picking, pred_mag = pred

        reduction = 'mean'

        # ======================== Picking ======================= #
        if opt.label_type == 'p':
            weights = torch.add(torch.mul(gt_picking[:, 0], opt.loss_weight), 1).to(device)
            picking_loss = F.binary_cross_entropy(weight=weights, input=pred_picking[:, :, 0].to(device), target=gt_picking[:, 0].type(torch.FloatTensor).to(device), reduction=reduction)
        elif opt.label_type == 'other':
            picking_loss = 0.0
            weights = torch.add(torch.mul(gt_picking[:, 0], opt.loss_weight), 1).to(device)
            for i in range(2):
                picking_loss += F.binary_cross_entropy(weight=weights, input=pred_picking[:, :, i].squeeze().to(device), target=gt_picking[:, i].type(torch.FloatTensor).to(device), reduction=reduction)
        elif opt.label_type == 'all':
            pred_detection, pred_p, pred_s = pred_picking

            pred = [pred_p, pred_s]
            gt_seg, gt_picking, gt_detection, gt_mag, distance = gt

            loss_weight = [0.6, 0.35, 0.05]       # P, S, detection
        
            # ground-truth -> 0: P-phase, 1: S-phase, 2: other label
            # detection
            # prediction -> 0: detection, 1: P-phase picking, 2: S-phase picking
            picking_loss = 0.0
            for i in range(3):
                if i == 0 or i == 1:
                    weights = torch.add(torch.mul(gt_picking[:, i], opt.loss_weight*4), 1).to(device)
                    picking_loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred[i].squeeze().to(device), target=gt_picking[:, i].type(torch.FloatTensor).to(device), reduction=reduction)
                else:
                    weights = torch.add(torch.mul(gt_detection[:, 0], opt.loss_weight*4), 1).to(device)
                    picking_loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred_detection.squeeze().to(device), target=gt_detection[:, 0].type(torch.FloatTensor).to(device), reduction=reduction)

        # ====================== Temporal segmentation ======================== #
        if opt.seg_proj_type != 'none':
            # temporal segmentation loss
            # prediction: (batch, wavelength, 1)
            # ground-truth: (batch, wavelength)
            weights = torch.add(torch.mul(gt_seg, opt.loss_weight*4), 1).to(device)
            segmentation_loss = F.binary_cross_entropy(input=pred_seg[:, :, 0], target=gt_seg.type(torch.FloatTensor).to(device), weight=weights, reduction=reduction)
            # print(f"picking: {picking_loss}, segmentation: {segmentation_loss}")
        else:
            segmentation_loss = 0.0

        # ====================== Magnitude estimation ======================== #
        weights = torch.max(gt_mag, dim=1).values
        # weights[weights==0] = -2
        # mag_gt[mag_gt==0] = -2
        
        gt_mag = weights.clone()
        weights = weights * opt.loss_weight
        weights[weights==0] = 1
        distance_mask = distance.clone()
        distance_mask[distance_mask <= 110] = 1
        distance_mask[distance_mask > 110] = 0
        
        tmp_loss = F.l1_loss(input=pred_mag.squeeze(), target=gt_mag.type(torch.FloatTensor).to(device), reduction='none')
        magnitude_loss = (tmp_loss * weights.to(device) * distance_mask.to(device)).mean()

        loss = opt.segmentation_ratio * segmentation_loss + opt.magnitude_ratio * magnitude_loss + (1-opt.segmentation_ratio-opt.magnitude_ratio) * picking_loss
        # print(f"mag_gt: {gt_mag}\nmag_pred: {pred_mag[:, 0, 0]}")
        # print(f"segmentation: {segmentation_loss}, 'magnitude: {magnitude_loss}, picking: {picking_loss}")
    elif opt.model_opt == 'real_GRADUATE' or opt.model_opt == 'real_GRADUATE_noNorm' or opt.model_opt == 'real_GRADUATE_noNorm_double':
        pred_seg, pred_mag, pred_picking = pred
        seg_gt, mag_gt, picking_gt, detection_gt, distance = gt
        
        loss_weight = [0.6, 0.35, 0.05]
        reduction = 'mean'

        # ground-truth -> 0: P-phase, 1: S-phase, 2: other label
        # detection
        # prediction -> 0: detection, 1: P-phase picking, 2: S-phase picking
        picking_loss = 0.0
        for i in range(3):
            if i == 0 or i == 1:
                nonzero_idx = (picking_gt[:, i] != 0)
                weights = torch.ones(picking_gt[:, i].shape)*0.11
                weights[nonzero_idx] = 0.89 if opt.loss_weight == -1 else opt.loss_weight 

                picking_loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred_picking[:, :, i+1].to(device), target=picking_gt[:, i].type(torch.FloatTensor).to(device), reduction=reduction)
            else:
                nonzero_idx = (detection_gt[:, 0] != 0)
                weights = torch.ones(detection_gt[:, 0].shape)*0.11
                weights[nonzero_idx] = 0.89 if opt.loss_weight == -1 else opt.loss_weight 

                picking_loss += loss_weight[i] * F.binary_cross_entropy(weight=weights.to(device), input=pred_picking[:, :, 0].to(device), target=detection_gt[:, 0].type(torch.FloatTensor).to(device), reduction=reduction)
        
        # temporal segmentation
        nonzero_idx = (seg_gt != 0)
        weights = torch.ones(seg_gt.shape)
        weights[nonzero_idx] = 30

        segmentation_loss = F.binary_cross_entropy(input=pred_seg[:, :, 0], target=seg_gt.type(torch.FloatTensor).to(device), weight=weights.to(device), reduction=reduction)
        
        # magnitude estimation
        weights = torch.max(mag_gt, dim=1).values
        # weights[weights==0] = -2
        # mag_gt[mag_gt==0] = -2
        
        mag_gt = weights.clone()
        weights = weights * 3
        weights[weights==0] = 1
        distance_mask = distance.clone()
        distance_mask[distance_mask <= 110] = 1
        distance_mask[distance_mask > 110] = 0
        
        tmp_loss = F.l1_loss(input=pred_mag.squeeze(), target=mag_gt.type(torch.FloatTensor).to(device), reduction='none')
        magnitude_loss = (tmp_loss * weights.to(device) * distance_mask.to(device)).mean()
        
        loss = opt.segmentation_ratio * segmentation_loss + opt.magnitude_ratio * magnitude_loss + (1-opt.segmentation_ratio-opt.magnitude_ratio) * picking_loss
        # print(f"mag_gt: {mag_gt}\nmag_pred: {pred_mag[:, 0, 0]}")
        # print(f"segmentation: {segmentation_loss}, 'magnitude: {magnitude_loss}, picking: {picking_loss}")
    elif opt.model_opt == 'ensemble_picker':
        # picking loss
        weights = torch.add(torch.mul(gt[:, 0], opt.loss_weight), 1).to(device)

        pred = pred.squeeze()
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)

        loss = F.binary_cross_entropy(weight=weights, input=pred.to(device), target=gt[:, 0].type(torch.FloatTensor).to(device))

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
