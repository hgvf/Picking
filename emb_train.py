from emb_model import *
from emb_loss import *
from modules import *
from optimizer import *
from wmseg_dataparallel import *

import seisbench.data as sbd
import seisbench.generate as sbg

import argparse
import os
import numpy as np
import logging
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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
    parser.add_argument('--clip_norm', type=float, default=0.1)
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

    # options
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--model_opt', type=str, default='none')
    parser.add_argument('--loss_weight', type=float, default=50)
    parser.add_argument('--dataset_opt', type=str, default='all')
    parser.add_argument('--loading_method', type=str, default='full')
    parser.add_argument('--normalize_opt', type=str, default='peak')
    parser.add_argument('--loss_opt', type=str, default='NT-Xent')
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--chunk_step', type=int, default=2)
    parser.add_argument('--mask_prob', type=int, default=0.3)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--mask_latent', type=bool, default=False)

    # custom hyperparameters
    parser.add_argument('--conformer_class', type=int, default=16)
    parser.add_argument('--d_ffn', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=3)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=1000)
    parser.add_argument('--encoder_type', type=str, default='conformer')
    parser.add_argument('--decoder_type', type=str, default='crossattn')
    parser.add_argument('--MGAN_normtype', type=str, default='mean')
    parser.add_argument('--MGAN_l', type=int, default=10)
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_class', type=int, default=128)

    # CPC hyperparameters
    parser.add_argument('--pred_timestep', type=int, default=10)
    parser.add_argument('--num_negative', type=int, default=30)

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

def load_dataset(opt):
    cwbsn, tsmip, stead = 0, 0, 0
    
    # loading datasets
    if opt.dataset_opt == 'stead' or opt.dataset_opt == 'all':
        # STEAD
        print('loading STEAD')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas3/earthquake_dataset_large/script/STEAD/'}}
        stead = sbd.STEAD(**kwargs)

    if opt.dataset_opt == 'cwbsn' or opt.dataset_opt == 'all':
        # CWBSN 
        print('loading CWBSN')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/CWBSN/seisbench/'}}

        cwbsn = sbd.CWBSN(loading_method=opt.loading_method, **kwargs)

        complete_mask = cwbsn.metadata['trace_completeness'] == 4
        single_mask = cwbsn.metadata['trace_number_of_event'] == 1
        mask = np.logical_and(single_mask, complete_mask)

        cwbsn.filter(mask)

    if opt.dataset_opt == 'tsmip' or opt.dataset_opt == 'all':
        # TSMIP
        print('loading TSMIP')
        kwargs={'download_kwargs': {'basepath': '/mnt/nas2/TSMIP/seisbench/seisbench/'}}

        tsmip = sbd.TSMIP(loading_method=opt.loading_method, sampling_rate=100, **kwargs)

        tsmip.metadata['trace_sampling_rate_hz'] = 100
        mask = tsmip.metadata['trace_completeness'] == 1

        tsmip.filter(mask)

    # if opt.dataset_opt == 'stead_noise' or opt.dataset_opt == 'all':
    #     # STEAD noise
    #     print('loading STEAD noise')
    #     kwargs={'download_kwargs': {'basepath': '/mnt/nas3/STEAD/'}}

    #     stead = sbd.STEAD_noise(**kwargs)
    #     print('traces: ', len(stead))

    return cwbsn, tsmip, stead
    # return cwbsn, tsmip

def split_dataset(opt):
    # load datasets
    print('loading datasets')

    # split datasets
    if opt.dataset_opt == 'all':
        # cwbsn, tsmip = load_dataset(opt)
        cwbsn, tsmip, stead = load_dataset(opt)
        
        cwbsn_train, cwbsn_dev, cwbsn_test = cwbsn.train_dev_test()
        tsmip_train, tsmip_dev, tsmip_test = tsmip.train_dev_test()
        stead_train, stead_dev, stead_test = stead.train_dev_test()

        # train = cwbsn_train + tsmip_train + cwbsn_dev + tsmip_dev
        # test = cwbsn_test + tsmip_test
        train = cwbsn_train + tsmip_train + cwbsn_dev + tsmip_dev + stead_train + stead_dev
        test = cwbsn_test + tsmip_test + stead_test

    print(f'total traces -> train: {len(train)}, dev: {len(test)}')

    return train, test

def set_generators(opt, data):
    generator = sbg.GenericGenerator(data)

    # set generator with or without augmentations
    phase_dict = ['trace_p_arrival_sample']

    if opt.loss_opt == 'TS2Vec' or opt.loss_opt == 'MLM' or opt.loss_opt == 'InfoNCE' or opt.loss_opt == 'TSTCC':
        augmentations = [
                sbg.RandomWindow(windowlen=3000, strategy="pad"),
                sbg.VtoA(),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass'),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.ChangeDtype(np.float32),
                sbg.SNR(),
            ]
    else:    
        augmentations = [
                sbg.WindowAroundSample(phase_dict, samples_before=3000, windowlen=6000, selection="first", strategy="pad"),
                sbg.FixedWindow(p0=2250, windowlen=3000, strategy='pad'),
                sbg.VtoA(),
                sbg.Filter(N=5, Wn=[1, 45], btype='bandpass'),
                sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std'),
                sbg.ChangeDtype(np.float32),
                sbg.SNR(),
            ]

    generator.add_augmentations(augmentations)

    if opt.aug:
        noise_generator = sbg.GaussianNoise(keep_ori=True)
        gap_generator = sbg.AddGap()
        
        generator.augmentation(noise_generator)

    return generator

def load_model(opt, device):
    assert opt.model_opt != None, "Choose one of the model in seisbench."

    if opt.loss_opt == 'MLM':
        if opt.model_opt == 'conformer':
            model = MLM(opt.chunk_size, opt.chunk_step, opt.mask_prob, opt.model_opt, 
                        True, False, *(opt.conformer_class, opt.d_model, opt.d_ffn, opt.nhead, opt.enc_layers))
        elif opt.model_opt == 'transformer':
            model = MLM(opt.chunk_size, opt.chunk_step, opt.mask_prob, opt.model_opt, 
                        True, False, *(opt.d_model, opt.nhead, opt.d_ffn, opt.enc_layers, opt.emb_dim))
    else:
        if opt.loss_opt == 'TS2Vec':
            # model = TS2Vec(d_model=opt.d_model, d_ffn=opt.d_ffn, d_out=opt.emb_dim, nhead=opt.nhead, n_layers=opt.enc_layers, dropout=opt.dropout, model_opt=opt.model_opt, emb_dim=opt.emb_dim, mask_latent=opt.mask_latent)
            model = Conformer_TS2Vec(d_model=opt.d_model, d_ffn=opt.d_ffn, nhead=opt.nhead, enc_layers=opt.enc_layers, emb_dim=opt.emb_dim)
        elif opt.loss_opt == 'CPC':
            # model = CPC_model(d_model=opt.d_model, emb_dim=opt.emb_dim, d_ffn=opt.d_ffn, model_opt=opt.model_opt, n_layers=opt.enc_layers, nhead=opt.nhead
            #                 , pred_timesteps=opt.pred_timestep, num_negatives=opt.num_negative)
            model = Conformer_CPC(d_model=opt.d_model, emb_dim=opt.emb_dim, d_ffn=opt.d_ffn, enc_layers=opt.enc_layers, nhead=opt.nhead
                            , pred_timesteps=opt.pred_timestep, num_negatives=opt.num_negative)
        elif opt.loss_opt == 'TSTCC':
            # model = TS_TCC_model(emb_dim=opt.emb_dim, d_ffn=opt.d_ffn, n_layers=opt.enc_layers, nhead=opt.nhead, pred_timesteps=opt.pred_timestep)
            model = Conformer_TSTCC(emb_dim=opt.emb_dim, d_ffn=opt.d_ffn, enc_layers=opt.enc_layers, nhead=opt.nhead, pred_timesteps=opt.pred_timestep, d_model=opt.d_model)
        elif opt.model_opt == 'conformer':
            model = ConformerEmb(d_model=opt.d_model, d_ffn=opt.d_ffn, nhead=opt.nhead, enc_layers=opt.enc_layers, emb_dim=opt.emb_dim, mask_latent=opt.mask_latent)
        elif opt.model_opt == 'transformer':
            model = TransformerEmb(d_model=opt.d_model, d_ffn=opt.d_ffn, d_out=opt.emb_dim, nhead=opt.nhead, n_layers=opt.enc_layers, dropout=opt.dropout, mask_latent=opt.mask_latent)
        elif opt.model_opt == 'contextAware':
            model = ContextAwareCase(d_model=opt.d_model, d_ffn=opt.d_ffn, d_out=opt.n_class, emb_dim=opt.emb_dim, nhead=opt.nhead, n_layers=opt.enc_layers, dropout=opt.dropout)
    
    return BalancedDataParallel(16, model.double(), dim=0).to(device)
    # return model.double().to(device)

def loss_fn(opt, emb, emb_snr, device, emb2=None, criterion=None):
    if opt.loss_opt == 'NT_Xent':
        if opt.aug:
            z2 = emb2
            loss = NT_Xent_aug(emb, z2, emb_snr.to(device), opt.batch_size, temperature=opt.temperature)
        else:
            loss = NT_Xent_fn(emb, emb_snr.to(device), opt.batch_size, temperature=opt.temperature)
    elif opt.loss_opt == 'TS2Vec':
        z2 = emb2
        loss = hierarchical_contrastive_loss(z1=emb, z2=z2)
    elif opt.loss_opt == 'CPC':
        pos, neg = emb2
        
        loss = 0.0
        for i in range(emb.shape[0]):
            loss += InfoNCE(emb[i], pos[i], neg)
    elif opt.loss_opt == 'TSTCC':
        loss = criterion(emb, emb2)

    return loss

def train(model, optimizer, dataloader, device, cur_epoch, opt, criterion=None):
    model.train()
    train_loss = 0.0
    
    train_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, (data) in train_loop:
        if not opt.aug:
            if opt.loss_opt == 'TS2Vec':
                out1, out2 = model(data['X'][:, :3, :-1].to(device))
            elif opt.loss_opt == 'CPC':
                pred, pos, neg = model(data['X'][:, :3, :-1].to(device))
            elif opt.loss_opt == 'TSTCC':
                nce, c_w, c_s = model(data['X'][:, :3, :-1].to(device))
            else:
                out = model(data['X'][:, :3, :-1].to(device))
        else:
            out = model(data['X'][:, :3, :-1].to(device))
            out2 = model(data['X'][:, 3:, :-1].to(device))
        
        if opt.loss_opt != 'MLM':
            if opt.aug:
                loss = loss_fn(opt, out, data['X'][:, 0, -1].to(device), device, (out2))
            else:
                if opt.loss_opt == 'TS2Vec':
                    loss = loss_fn(opt, out1, out1, device, out2)
                    # loss = hierarchical_contrastive_loss(out1, out2)
                elif opt.loss_opt == 'CPC':
                    loss = loss_fn(opt, pred, 1, device, (pos, neg))
                elif opt.loss_opt == 'TSTCC':
                    lambda1, lambda2 = 1, 0.7

                    loss_cc = loss_fn(opt, c_w, 1, device, c_s, criterion)
                    loss = torch.mean(nce) * lambda1 + loss_cc * lambda2
                else:
                    loss = loss_fn(opt, out, data['X'][:, 0, -1].to(device), device)
        else:
            loss = out
            loss = torch.mean(loss)

        loss = loss / opt.gradient_accumulation
        loss.backward()
        
        if ((idx+1) % opt.gradient_accumulation == 0) or ((idx+1) == len(dataloader)):
            nn.utils.clip_grad_norm_(model.parameters(), opt.clip_norm)
            optimizer.step()

            model.zero_grad()
            # optimizer.zero_grad()
        
        train_loss = train_loss + loss.detach().cpu().item()*opt.gradient_accumulation
        train_loop.set_description(f"[Train Epoch {cur_epoch+1}/{opt.epochs}]")
        train_loop.set_postfix(loss=loss.detach().cpu().item()*opt.gradient_accumulation)
        
    train_loss = train_loss / (len(dataloader))
    return train_loss

def valid(model, dataloader, device, cur_epoch, opt, criterion=None):
    model.eval()
    dev_loss = 0.0

    valid_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, data in valid_loop:
        with torch.no_grad():
            if not opt.aug:
                if opt.loss_opt == 'TS2Vec':
                    out1, out2 = model(data['X'][:, :3, :-1].to(device))
                elif opt.loss_opt == 'CPC':
                    pred, pos, neg = model(data['X'][:, :3, :-1].to(device))
                elif opt.loss_opt == 'TSTCC':
                    nce, c_w, c_s = model(data['X'][:, :3, :-1].to(device))
                else:
                    out = model(data['X'][:, :3, :-1].to(device))
            else:
                out = model(data['X'][:, :3, :-1].to(device))
                out2 = model(data['X'][:, 3:, :-1].to(device))
            
            if opt.loss_opt != 'MLM':
                if opt.aug:
                    loss = loss_fn(opt, out, data['X'][:, 0, -1].to(device), device, (out2))
                else:
                    if opt.loss_opt == 'TS2Vec':
                        loss = loss_fn(opt, out1, out1, device, (out2))
                        # loss = hierarchical_contrastive_loss(out1, out2)
                    elif opt.loss_opt == 'CPC':
                        loss = loss_fn(opt, pred, 1, device, (pos, neg))
                    elif opt.loss_opt == 'TSTCC':
                        lambda1, lambda2 = 1, 0.7

                        loss_cc = loss_fn(opt, c_w, 1, device, c_s, criterion)
                        loss = torch.mean(nce) * lambda1 + loss_cc * lambda2
                    else:
                        loss = loss_fn(opt, out, data['X'][:, 0, -1].to(device), device)
            else:
                loss = out
                loss = torch.mean(loss)
        
        dev_loss = dev_loss + loss.detach().cpu().item()
        
        valid_loop.set_description(f"[Valid Epoch {cur_epoch+1}/{opt.epochs}]")
        valid_loop.set_postfix(loss=loss.detach().cpu().item())
        
    valid_loss = dev_loss / (len(dataloader))

    return valid_loss

if __name__ == '__main__':
    opt = parse_args()

    output_dir = os.path.join('./results_emb', opt.save_path)
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

    print('Save path: ', opt.save_path)
    logging.info('save path: %s' %(opt.save_path))
    logging.info('device: %s'%(device))

    # load model
    model = load_model(opt, device)

    if opt.load_pretrained:
        model_dir = os.path.join('./results_emb', opt.pretrained_path)
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

    train_set, test_set = split_dataset(opt)
    
    train_generator = set_generators(opt, train_set)
    test_generator = set_generators(opt, test_set)
   
    # create dataloaders
    print('creating dataloaders')
    train_loader = DataLoader(train_generator, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)
    test_loader = DataLoader(test_generator, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)
    logging.info('train: %d, dev: %d' %(len(train_loader)*opt.batch_size, len(test_loader)*opt.batch_size))

    # create criterion if TSTCC
    if opt.loss_opt == 'TSTCC':
        nt_xent_criterion = NTXentLoss(device, opt.batch_size, opt.temperature, True)
    else:
        nt_xent_criterion = None

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
        # min_loss = checkpoint['min_loss']
        min_loss = 10000
        logging.info('resume training at epoch: %d' %(init_epoch))
    else:
        init_epoch = 0
        min_loss = 100000

    # start training
    for epoch in range(init_epoch, opt.epochs):
        train_loss = train(model, optimizer, train_loader, device, epoch, opt, nt_xent_criterion)
        valid_loss = valid(model, test_loader, device, epoch, opt, nt_xent_criterion)

        print('[Train] epoch: %d -> loss: %.4f' %(epoch+1, train_loss))
        print('[Eval] epoch: %d -> loss: %.4f' %(epoch+1, valid_loss))
        print('Learning rate: %.10f' %(optimizer.learning_rate))
        logging.info('[Train] epoch: %d -> loss: %.4f' %(epoch+1, train_loss))
        logging.info('[Eval] epoch: %d -> loss: %.4f' %(epoch+1, valid_loss))
        logging.info('Learning rate: %.10f' %(optimizer.learning_rate))
        logging.info('======================================================')

        # Line notify
        toLine(opt.save_path, train_loss, valid_loss, epoch, opt.epochs, False)

        # Early stopping
        if valid_loss < min_loss:
            min_loss = valid_loss

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
