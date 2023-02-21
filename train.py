import argparse
import os
import numpy as np
import logging
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append('../RED-PAN')
from gen_tar import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from optimizer import *
from utils import *

import seisbench.data as sbd
import seisbench.generate as sbg

def parse_args():
    parser = argparse.ArgumentParser()
    
    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--recover_from_best', type=bool, default=False)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--clip_norm', type=float, default=0.01)
    parser.add_argument('--patience', type=float, default=7)
    parser.add_argument('--noam', type=bool, default=False)
    parser.add_argument('--warmup_step', type=int, default=1500)
    parser.add_argument('--load_pretrained', type=bool, default=False)
    parser.add_argument('--pretrained_path', type=str, default='tmp')
    parser.add_argument('--model_name', type=str, default='checkpoint_last')

    # save_path
    parser.add_argument("--save_path", type=str, default='tmp')
    parser.add_argument("--valid_step", type=int, default=3000)
    parser.add_argument('--valid_on_training', type=bool, default=False)

    # dataset hyperparameters
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--snr_threshold', type=float, default=-1)
    parser.add_argument('--snr_schedule', type=bool, default=False)
    parser.add_argument('--level_schedule', type=bool, default=False)
    parser.add_argument('--level', type=int, default=-1)
    parser.add_argument('--schedule_patience', type=int, default=5)
    parser.add_argument('--init_stage', type=int, default=0)
    parser.add_argument('--s_wave', type=bool, default=False)

    # data augmentations
    parser.add_argument('--gaussian_noise_prob', type=float, default=0.5)
    parser.add_argument('--channel_dropout_prob', type=float, default=0.3)
    parser.add_argument('--adding_gap_prob', type=float, default=0.2)
    parser.add_argument('--shift_to_end_prob', type=float, default=0.0)
    parser.add_argument('--mask_afterP', type=float, default=0.0)

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
    parser.add_argument('--query_type', type=str, default='pos_emb')
    parser.add_argument('--intensity_MT', type=bool, default=False)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

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
    parser.add_argument('--cross_attn_type', type=int, default=1)
    parser.add_argument('--n_segmentation', type=int, default=4)
    parser.add_argument('--output_layer_type', type=str, default='fc')
    parser.add_argument('--rep_KV', type=str, default='False')
    parser.add_argument('--segmentation_ratio', type=float, default=0.3)
    parser.add_argument('--seg_proj_type', type=str, default='crossattn')

    opt = parser.parse_args()

    return opt

def toLine(save_path, train_loss, valid_loss, epoch, n_epochs, isFinish):
    token = "Eh3tinCwQ87qfqD9Dboy1mpd9uMavhGV9u5ohACgmCF"

    if not isFinish:
        message = save_path + ' -> Epoch ['+ str(epoch) + '/' + str(n_epochs) + '] train_loss: ' + str(train_loss) +', valid_loss: ' +str(valid_loss)
    else:
        message = save_path + ' -> Finish training...'

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

def noam_optimizer(model, lr, warmup_step, device):
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    o = {}
    o["method"] = 'adam'
    o['lr'] = lr
    o['max_grad_norm'] = 0
    o['beta1'] = 0.9
    o['beta2'] = 0.999
    o['decay_method'] = 'noam'
    o['warmup_steps'] = warmup_step
    optimizer = build_optim(o, model, False, device)

    return optimizer

def split_dataset(opt, return_dataset=False):
    # load datasets
    print('loading datasets')
    cwbsn, tsmip, stead, cwbsn_noise = load_dataset(opt)

    # split datasets
    if opt.dataset_opt == 'all':
        cwbsn_train, cwbsn_dev, _ = cwbsn.train_dev_test()
        tsmip_train, tsmip_dev, _ = tsmip.train_dev_test()
        stead_train, stead_dev, _ = stead.train_dev_test()
        cwbsn_noise_train, cwbsn_noise_dev, _ = cwbsn_noise.train_dev_test()

        if opt.dataset_opt == 'redpan':
            stead_train.filter(stead_train.metadata.index <= 76500)

        train = cwbsn_train + tsmip_train + stead_train + cwbsn_noise_train
        dev = cwbsn_dev + tsmip_dev + stead_dev + cwbsn_noise_dev
    if opt.dataset_opt == 'taiwan' or opt.dataset_opt == 'redpan':        
        cwbsn_train, cwbsn_dev, _ = cwbsn.train_dev_test()
        tsmip_train, tsmip_dev, _ = tsmip.train_dev_test()
        cwbsn_noise_train, cwbsn_noise_dev, _ = cwbsn_noise.train_dev_test()

        train = cwbsn_train + tsmip_train + cwbsn_noise_train
        dev = cwbsn_dev + tsmip_dev + cwbsn_noise_dev
    elif opt.dataset_opt == 'cwbsn':
        train, dev, _ = cwbsn.train_dev_test()
    elif opt.dataset_opt == 'tsmip':
        train, dev, _ = tsmip.train_dev_test()
    elif opt.dataset_opt == 'stead':
        train, dev, _ = stead.train_dev_test()
    elif opt.dataset_opt == 'prev_taiwan':
        cwbsn_train, cwbsn_dev, _ = cwbsn.train_dev_test()
        tsmip_train, tsmip_dev, _ = tsmip.train_dev_test()
        stead_train, stead_dev, _ = stead.train_dev_test()

        train = cwbsn_train + tsmip_train + stead_train
        dev = cwbsn_dev + tsmip_dev + stead_dev

    print(f'total traces -> train: {len(train)}, dev: {len(dev)}')

    if not return_dataset:
        return train, dev
    else:
        return train, dev, cwbsn, tsmip, stead

def set_generators(opt, data):
    generator = sbg.GenericGenerator(data)

    # set generator with or without augmentations
    phase_dict = ['trace_p_arrival_sample']
    augmentations = basic_augmentations(opt, phase_dict)

    generator.add_augmentations(augmentations)

    if opt.aug and not opt.model_opt == 'RED_PAN':
        # data augmentation during training
        # 1) Add gaps (0.2)
        # 2) Channel dropout (0.3)
        # 3) Gaussian noise (0.5)
        # 4) Mask AfterP (0.3)
        # 5) Shift to end (0.2)

        gap_generator = sbg.OneOf([sbg.AddGap(), sbg.NullAugmentation()], [opt.adding_gap_prob, 1-opt.adding_gap_prob])
        dropout_generator = sbg.OneOf([sbg.ChannelDropout(), sbg.NullAugmentation()], [opt.channel_dropout_prob, 1-opt.channel_dropout_prob])
        noise_generator = sbg.OneOf([sbg.GaussianNoise(), sbg.NullAugmentation()], [opt.gaussian_noise_prob, 1-opt.gaussian_noise_prob])
        mask_afterP_generator = sbg.OneOf([sbg.MaskafterP(), sbg.NullAugmentation()], [opt.mask_afterP, 1-opt.mask_afterP])
        shift_generator = sbg.OneOf([sbg.ShiftToEnd(), sbg.NullAugmentation()], [opt.shift_to_end_prob, 1-opt.shift_to_end_prob])

        generator.augmentation(gap_generator)
        generator.augmentation(dropout_generator)
        generator.augmentation(noise_generator)
        generator.augmentation(mask_afterP_generator)
        generator.augmentation(shift_generator)

    return generator

def snr_scheduler(cwbsn, tsmip, stead, stage):
    all = [cwbsn, tsmip]
    snr_stage = [25, 15, 7, 0, -15, -100]

    if stage >= len(snr_stage)-1:
        stage = len(snr_stage)-1

    snr_range = snr_stage[stage]
    for data in all:    
        mask = data.metadata['trace_Z_snr_db'] >= snr_range
        data.filter(mask)

    cwbsn_train, cwbsn_dev, _ = cwbsn.train_dev_test()
    tsmip_train, tsmip_dev, _ = tsmip.train_dev_test()
    stead_train, stead_dev, _ = stead.train_dev_test()

    train = cwbsn_train + tsmip_train + stead_train
    dev = cwbsn_dev + tsmip_dev + stead_dev

    print(f'snr_schedule: train -> {len(train)}, dev -> {len(dev)}')

    train_generator = set_generators(opt, train)
    dev_generator = set_generators(opt, dev)

    return train_generator, dev_generator, snr_range

def snr_stage(prev_loss, cur_loss, prev_stage, schedule_cnt, patience):
    if prev_loss == 1000:
        return prev_stage, schedule_cnt, False
    else:
        if cur_loss >= prev_loss:
            schedule_cnt += 1
        else:
            return prev_stage, 0, False

    if schedule_cnt == patience:
        schedule_cnt = 0
        print('move to next stage....')
        return prev_stage+1, schedule_cnt, True
    else:
        return prev_stage, schedule_cnt, False

def level_scheduler(cwbsn, tsmip, stead, stage):
    if stage >= 1:
        stage = 1

    if stage == 0:
        mask = cwbsn.metadata['trace_completeness'] == 4

    elif stage == 1:
        mask = np.logical_or(cwbsn.metadata['trace_completeness'] == 3, cwbsn.metadata['trace_completeness'] == 4)
    
    cwbsn.filter(mask)

    cwbsn_train, cwbsn_dev, _ = cwbsn.train_dev_test()
    tsmip_train, tsmip_dev, _ = tsmip.train_dev_test()
    stead_train, stead_dev, _ = stead.train_dev_test()

    train = cwbsn_train + tsmip_train + stead_train
    dev = cwbsn_dev + tsmip_dev + stead_dev

    print(f'snr_schedule: train -> {len(train)}, dev -> {len(dev)}')

    train_generator = set_generators(opt, train)
    dev_generator = set_generators(opt, dev)

    return train_generator, dev_generator

def level_stage(prev_loss, cur_loss, prev_stage, schedule_cnt, patience):
    if prev_loss == 1000:
        return prev_stage, schedule_cnt, False
    else:
        if cur_loss >= prev_loss:
            schedule_cnt += 1
        else:
            return prev_stage, 0, False

    if schedule_cnt == patience:
        schedule_cnt = 0
        print('move to next stage....')

        return prev_stage+1, schedule_cnt, True
    else:
        return prev_stage, schedule_cnt, False

def train(model, optimizer, dataloader, valid_loader, device, cur_epoch, opt, output_dir, redpan_loss=None):
    model.train()
    train_loss = 0.0
    min_loss = 1000
    task1_loss, task2_loss = 0.0, 0.0
    import matplotlib.pyplot as plt
    train_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, (data) in train_loop:
        if opt.model_opt == 'basicphaseAE':
            data['X'] = data['X'].reshape(-1, 3, 600)
        
        if opt.model_opt == 'RED_PAN':
            if opt.aug:
                out = model(REDPAN_aug(data['X'])[:, :3].to(device))
            else:
                out = model(data['X'][:, :3].to(device))
            
            loss, PS_loss, M_loss = loss_fn(opt, out, (data['X'][:, 3:6], data['X'][:, 6:]), device, redpan_loss, cur_epoch)
            task1_loss += PS_loss.cpu().item()
            task2_loss += M_loss.cpu().item()
        else:
            if opt.model_opt == 'conformer_stft':
                out = model(data['X'].to(device), stft=data['stft'].to(device).float())
            elif opt.model_opt == 'conformer_intensity':
                out, out_MT = model(data['X'].to(device))
            elif opt.model_opt == 'GRADUATE':
                # plt.subplot(211)
                # plt.plot(data['X'][0][:3].T)
                # plt.subplot(212)
                # plt.plot(data['X'][1][:3].T)
                # plt.savefig(f"{idx}.png")
                # plt.clf()
                out_seg, out = model(data['X'].to(device), stft=data['stft'].to(device).float())

            else:
                out = model(data['X'].to(device))
            
            if opt.model_opt == 'conformer_intensity':
                loss = loss_fn(opt, out, data['y'], device, intensity=(out_MT, data['intensity']))
            elif opt.model_opt == 'GRADUATE':
                loss = loss_fn(opt, pred=(out_seg, out), gt=(data['seg'], data['y']), device=device)
            else:
                loss = loss_fn(opt, out, data['y'], device)
    
        loss = loss / opt.gradient_accumulation
        loss.backward()
  
        if ((idx+1) % opt.gradient_accumulation == 0) or ((idx+1) == len(dataloader)):
            if opt.clip_norm != 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), opt.clip_norm)
            optimizer.step()
            
            model.zero_grad()
            # optimizer.zero_grad()
        
        train_loss = train_loss + loss.detach().cpu().item()*opt.gradient_accumulation
        train_loop.set_description(f"[Train Epoch {cur_epoch+1}/{opt.epochs}]")
        train_loop.set_postfix(loss=loss.detach().cpu().item()*opt.gradient_accumulation)
        
    train_loss = train_loss / (len(dataloader))
    
    if opt.model_opt == 'RED_PAN':
        return (train_loss, task1_loss / (len(dataloader)), task2_loss / (len(dataloader)))
    else:
        return train_loss

def valid(model, dataloader, device, cur_epoch, opt, redpan_loss=None):
    model.eval()
    dev_loss = 0.0
    task1_loss, task2_loss = 0.0, 0.0

    valid_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, data in valid_loop:
        with torch.no_grad():
            if opt.model_opt == 'basicphaseAE':
                data['X'] = data['X'].reshape(-1, 3, 600)
            
            if opt.model_opt == 'RED_PAN':
                if opt.aug:
                    out = model(REDPAN_aug(data['X'])[:, :3].to(device))
                else:
                    out = model(data['X'][:, :3].to(device))
                
                loss, PS_loss, M_loss = loss_fn(opt, out, (data['X'][:, 3:6], data['X'][:, 6:]), device, redpan_loss, cur_epoch)
                task1_loss += PS_loss.cpu().item()
                task2_loss += M_loss.cpu().item()
            else:
                if opt.model_opt == 'conformer_stft':
                    out = model(data['X'].to(device), stft=data['stft'].to(device).float())
                elif opt.model_opt == 'conformer_intensity':
                    out, out_MT = model(data['X'].to(device))
                elif opt.model_opt == 'GRADUATE':
                    out_seg, out = model(data['X'].to(device), stft=data['stft'].to(device).float())
                else:
                    out = model(data['X'].to(device))

                if opt.model_opt == 'conformer_intensity':
                    loss = loss_fn(opt, out, data['y'], device, intensity=(out_MT, data['intensity']))
                elif opt.model_opt == 'GRADUATE':
                    loss = loss_fn(opt, pred=(out_seg, out), gt=(data['seg'], data['y']), device=device)
                else:
                    loss = loss_fn(opt, out, data['y'], device)
            
        dev_loss = dev_loss + loss.detach().cpu().item()

        valid_loop.set_description(f"[Valid Epoch {cur_epoch+1}/{opt.epochs}]")
        valid_loop.set_postfix(loss=loss.detach().cpu().item())
        
    valid_loss = dev_loss / (len(dataloader))

    return valid_loss

if __name__ == '__main__':
    opt = parse_args()

    output_dir = os.path.join('./results', opt.save_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    log_path = os.path.join(output_dir, 'train.log')
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                        filename=log_path, 
                        filemode='a', 
                        level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S",)
    logging.info('start training')
    logging.info('configs: ')
    logging.info(opt)
    logging.info('======================================================')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    os.environ["OMP_NUM_THREADS"] = str(opt.workers)

    logging.info('device: %s'%(device))

    # load model
    if opt.pretrained_emb is not None:
        logging.info('loading pretrained embedding: %s' %(opt.pretrained_emb))

    model = load_model(opt, device)
    
    if opt.load_pretrained:
        model_dir = os.path.join('./results', opt.pretrained_path)
        model_path = os.path.join(model_dir, 'model.pt')
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info('trainable parameters: %d' %(trainable))
    print('trainable parameters: %d' %(trainable))

    print('loading optimizer & scheduler...')
    if opt.noam:
        optimizer = noam_optimizer(model, opt.lr, opt.warmup_step, device)
    else:
        optimizer = optim.Adam(model.parameters(), opt.lr)

    if opt.snr_schedule or opt.level_schedule:
        train_set, dev_set, cwbsn, tsmip, stead = split_dataset(opt, return_dataset=True)
    else:
        train_set, dev_set = split_dataset(opt, return_dataset=False)
    
        train_generator = set_generators(opt, train_set)
        dev_generator = set_generators(opt, dev_set)

        # create dataloaders
        print('creating dataloaders')
        train_loader = DataLoader(train_generator, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        dev_loader = DataLoader(dev_generator, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        logging.info('train: %d, dev: %d' %(len(train_loader)*opt.batch_size, len(dev_loader)*opt.batch_size))

    # load checkpoint
    if opt.resume_training:
        logging.info('Resume training...')

        if opt.recover_from_best:
            checkpoint = torch.load(os.path.join(output_dir, 'model.pt'), map_location=device)
            print('resume training, load model.pt...')
        else:
            checkpoint = torch.load(os.path.join(output_dir, 'checkpoint_last.pt'), map_location=device)
            print('resume training, load checkpoint_last.pt...')

        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        min_loss = checkpoint['min_loss']
        print(f'start from epoch {init_epoch}, min_loss: {min_loss}')
        logging.info('resume training at epoch: %d' %(init_epoch))
    else:
        init_epoch = 0
        min_loss = 100000
    
    # RED_PAN: record loss for every epoch
    if opt.model_opt == 'RED_PAN':
        PS_loss, M_loss = [], []

    stage, early_stop_cnt, schedule_cnt = opt.init_stage, 0, 0
    prev_loss, valid_loss = 1000, 1000
    isNext = False
    for epoch in range(init_epoch, opt.epochs):

        # SNR schedule learning
        if opt.snr_schedule or opt.level_schedule:
            if not isNext:
                if valid_loss < prev_loss:
                    prev_loss = valid_loss 
            else:
                min_loss = 1000
                prev_loss = 1000

            if isNext or epoch == 0 or opt.resume_training:
                if opt.snr_schedule:
                    train_generator, dev_generator, snr_range = snr_scheduler(cwbsn.copy(), tsmip.copy(), stead.copy(), stage)
                    logging.info('snr scheduler learning, snr >= %f' %(snr_range))
                elif opt.level_schedule:
                    train_generator, dev_generator = level_scheduler(cwbsn.copy(), tsmip.copy(), stead.copy(), stage)

                train_loader = DataLoader(train_generator, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
                dev_loader = DataLoader(dev_generator, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
                logging.info('train: %d, dev: %d' %(len(train_loader)*opt.batch_size, len(dev_loader)*opt.batch_size))

        if opt.model_opt == 'RED_PAN':
            train_loss = train(model, optimizer, train_loader, dev_loader, device, epoch, opt, output_dir, redpan_loss=(PS_loss, M_loss))
            valid_loss = valid(model, dev_loader, device, epoch, opt, redpan_loss=(PS_loss, M_loss))
        else:
            train_loss = train(model, optimizer, train_loader, dev_loader, device, epoch, opt, output_dir)
            valid_loss = valid(model, dev_loader, device, epoch, opt)

        if opt.model_opt == 'RED_PAN':
            train_loss, task1_loss, task2_loss = train_loss
            PS_loss.append(task1_loss)
            M_loss.append(task2_loss)
        if opt.snr_schedule:
            stage, schedule_cnt, isNext = snr_stage(prev_loss, valid_loss, stage, schedule_cnt, opt.schedule_patience)
        
        if opt.level_schedule:
            stage, schedule_cnt, isNext = level_stage(prev_loss, valid_loss, stage, schedule_cnt, opt.schedule_patience)

        print('[Train] epoch: %d -> loss: %.4f' %(epoch+1, train_loss))
        print('[Eval] epoch: %d -> loss: %.4f' %(epoch+1, valid_loss))
        logging.info('[Train] epoch: %d -> loss: %.4f' %(epoch+1, train_loss))
        logging.info('[Eval] epoch: %d -> loss: %.4f' %(epoch+1, valid_loss))

        if opt.noam:
            print('Learning rate: %.10f' %(optimizer.learning_rate))
            logging.info('Learning rate: %.10f' %(optimizer.learning_rate))
        logging.info('======================================================')

        # Line notify
        toLine(opt.save_path, train_loss, valid_loss, epoch, opt.epochs, False)

        # Early stopping
        if valid_loss < min_loss:
            min_loss = valid_loss

            # Saving model
            if opt.snr_schedule or opt.level_schedule:
                targetPath = os.path.join(output_dir, 'model_' + str(stage) + '.pt')
            else:
                targetPath = os.path.join(output_dir, 'model.pt')

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer,
                'min_loss': min_loss,
                'epoch': epoch
            }, targetPath)

            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        # Saving model
        targetPath = os.path.join(output_dir, opt.model_name+'.pt')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer,
            'min_loss': min_loss,
            'epoch': epoch
        }, targetPath)

        if early_stop_cnt == opt.patience:
            logging.info('early stopping...')

            break

    print('Finish training...')
    toLine(opt.save_path, train_loss, valid_loss, epoch, opt.epochs, True)













