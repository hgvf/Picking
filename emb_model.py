import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import *
from modules import *

import numpy as np
import math
import random

# ============== Embedding before encoder & decoder ============== #
class TransformerEmb(nn.Module):
    def __init__(self, d_model, d_ffn, d_out, nhead, n_layers, dropout=0.1, mask_latent=False):
        super(TransformerEmb, self).__init__()
        
        self.mask_latent = mask_latent

        self.subsample = nn.Sequential(
            nn.Conv1d(3, d_out, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(d_out, d_out, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)
        
        self.transformer = nn.TransformerEncoderLayer(d_model=d_out, nhead=nhead, dim_feedforward=d_ffn, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=n_layers)

    def forward(self, wave):
        # wave: (batch, 3, 3000)
    
        wave = self.subsample(wave).permute(0,2,1)

        if self.mask_latent:
            mask = generate_binomial_mask(wave.size(0), wave.size(1), wave.shape[2]).to(wave.device)
            tmp_wave = wave.clone()
            tmp_wave[~mask] = 0
            wave = tmp_wave

        wave = self.dropout(wave)
        out = self.encoder(wave)

        return out

class ConformerEmb(nn.Module):
    def __init__(self, d_model, d_ffn, nhead, enc_layers, emb_dim, mask_latent=False):
        super(ConformerEmb, self).__init__()

        self.mask_latent = mask_latent

        self.batchnorm = nn.BatchNorm1d(d_model)
        self.subsample = nn.Sequential(
                nn.Conv1d(3, d_out, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(d_out, d_out, kernel_size=3, stride=2),
                nn.ReLU(),
            )
        self.encoder = Conformer(num_classes=emb_dim, input_dim=d_out, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=enc_layers, subsample=False)

    def forward(self, wave):
        wave = self.batchnorm(wave)
        
        wave = self.subsample(wave).permute(0,2,1)

        if self.mask_latent:
            mask = generate_binomial_mask(wave.size(0), wave.size(1), wave.shape[2]).to(wave.device)
            tmp_wave = wave.clone()
            tmp_wave[~mask] = 0
            wave = tmp_wave

        emb, _ = self.encoder(wave, 3000)

        return emb

class ContextAwareCase(nn.Module):
    def __init__(self, d_model, d_ffn, d_out, emb_dim, nhead, n_layers, dropout=0.1):
        super(ContextAwareCase, self).__init__()
        
        self.dialted_conv = nn.Sequential(nn.Conv1d(d_model, 8, dilation=2, kernel_size=5, padding='same'),
                                          nn.ReLU(),
                                          nn.Conv1d(8, 16, dilation=3, kernel_size=5, padding='same'),
                                          nn.ReLU(),
                                          nn.Conv1d(16, d_out, dilation=3, kernel_size=5, padding='same'),
                                          nn.ReLU())
        
        self.dropout = nn.Dropout(dropout)
        
        self.transformer = nn.TransformerEncoderLayer(d_model=d_out, nhead=nhead, dim_feedforward=d_ffn, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=n_layers)
        
        self.softmax = nn.Softmax(dim=-1)
        self.embedding = nn.Embedding(d_out, emb_dim)
        
        nn.init.kaiming_uniform_(self.embedding.weight)
    def forward(self, wave):
        # wave: (batch, 3, 3000)
        wave = self.dialted_conv(wave).permute(0,2,1)
        print('dialted_conv: ', wave)
        wave = self.dropout(wave)
        out = self.softmax(self.encoder(wave))
        print('softmaxed encoder: ', out)
        
        idx = torch.argmax(out, dim=-1)
        emb = self.embedding(idx)
        print('emb: ', emb)
        return emb

class MLM(nn.Module):
    def __init__(self, chunk_size, chunk_step, mask_prob, model_opt, return_loss=True, inference=False, *args):
        super(MLM, self).__init__()
        
        self.return_loss = return_loss
        self.chunk_size = chunk_size
        self.chunk_step = chunk_step
        self.mask_prob = mask_prob
        self.model_opt = model_opt
        self.inference = inference
        self.dec_layers = 4

        if model_opt == 'conformer':
            num_classes, d_model, d_ffn, nhead, n_layers = args

            self.subsample = nn.Sequential(
                nn.Conv1d(d_model, emb_dim, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(emb_dim, emb_dim, kernel_size=3, stride=2),
                nn.ReLU(),
            )
            
            self.model = Conformer(num_classes=num_classes, input_dim=emb_dim, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=n_layers, subsample=False)

            self.prediction_head = nn.Sequential(nn.Linear(num_classes, 64),
                                            nn.GELU(),
                                            nn.Linear(64, 32),
                                            nn.GELU(),
                                            nn.Linear(32, 16),
                                            nn.GELU(),
                                            nn.Linear(16, d_model))

        elif model_opt == 'transformer':
            d_model, nhead, d_ffn, n_layers, emb_dim = args
        
            self.subsample = nn.Sequential(
                nn.Conv1d(d_model, emb_dim, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(emb_dim, emb_dim, kernel_size=3, stride=2),
                nn.ReLU(),
            )

            self.transformer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dim_feedforward=d_ffn, batch_first=True)
            self.model = nn.TransformerEncoder(self.transformer, num_layers=n_layers)
            
        self.crossAttnLayer = nn.ModuleList([cross_attn_layer(nhead, emb_dim//nhead, emb_dim//nhead, d_model, emb_dim, d_ffn)
                                                for _ in range(self.dec_layers)]
                                                )
        self.pos_emb = PositionalEncoding(emb_dim, max_len=3000, return_vec=True)
        
        if not self.inference:
            self.prediction_head = nn.Linear(emb_dim, 3)

    def forward(self, x):
        # x: (batch, 3, wave_length)
        
        if not self.inference:
            masked_x, mask_Z, mask_N, mask_E, n_chunks = self.masking(x, self.chunk_size, self.chunk_step, self.mask_prob)
        else:
            masked_x = x

        masked_x = self.subsample(masked_x).permute(0,2,1)

        if self.model_opt == 'conformer':
            out, _ = self.model(masked_x.clone().detach(), x.shape[1])
        elif self.model_opt == 'transformer':
            out = self.model(masked_x.clone().detach())

        pos_emb = self.pos_emb(x).unsqueeze(0).repeat(x.size(0), 1, 1)
            
        for i in range(self.dec_layers):
            if i == 0:
                dec_out = self.crossAttnLayer[i](pos_emb, out, out)
            else:
                dec_out = self.crossAttnLayer[i](dec_out, out, out)

        if self.return_loss:
            out = self.prediction_head(dec_out)
            loss = self.loss_fn(x.permute(0,2,1), out, mask_Z, mask_N, mask_E, self.chunk_size, self.chunk_step)
        
            return loss
        else:
            return dec_out
    
    def prob_mask_like(self, t, prob):
        return torch.zeros_like(t).float().uniform_(0, 1) < prob
    
    def masking(self, x, chunk_size, chunk_step, mask_prob):
        n_chunks = x.shape[1] // chunk_size // chunk_step
        chunk_idx = torch.zeros(n_chunks)

        mask_Z = self.prob_mask_like(chunk_idx, mask_prob)
        mask_N = self.prob_mask_like(chunk_idx, mask_prob)
        mask_E = self.prob_mask_like(chunk_idx, mask_prob)
        
        mask_Z = [idx for idx, i in enumerate(mask_Z) if torch.any(i)]
        mask_N = [idx for idx, i in enumerate(mask_N) if torch.any(i)]
        mask_E = [idx for idx, i in enumerate(mask_E) if torch.any(i)]
        
        for m in mask_Z:
            x[:, m*chunk_step*chunk_size:m*chunk_step*chunk_size+chunk_size, 0] = 0
        for m in mask_N:
            x[:, m*chunk_step*chunk_size:m*chunk_step*chunk_size+chunk_size, 1] = 0
        for m in mask_E:
            x[:, m*chunk_step*chunk_size:m*chunk_step*chunk_size+chunk_size, 2] = 0
            
        return x, mask_Z, mask_N, mask_E, n_chunks
    
    def loss_fn(self, y, x, mask_Z, mask_N, mask_E, chunk_size, chunk_step):
        mask_Z, mask_N, mask_E = torch.tensor(mask_Z), torch.tensor(mask_N), torch.tensor(mask_E)
        
        n_chunks = x.shape[1] // chunk_size // chunk_step
        chunk_idx = torch.arange(n_chunks)

        Z_compareview = mask_Z.repeat(chunk_idx.shape[0],1).T
        N_compareview = mask_N.repeat(chunk_idx.shape[0],1).T
        E_compareview = mask_E.repeat(chunk_idx.shape[0],1).T

        Z_unmask = chunk_idx[(Z_compareview != chunk_idx).T.prod(1)==1]
        N_unmask = chunk_idx[(N_compareview != chunk_idx).T.prod(1)==1]
        E_unmask = chunk_idx[(E_compareview != chunk_idx).T.prod(1)==1]
        
        for m in Z_unmask:
            x[:, m*chunk_step*chunk_size:m*chunk_step*chunk_size+chunk_size, 0] = 0
            y[:, m*chunk_step*chunk_size:m*chunk_step*chunk_size+chunk_size, 0] = 0
        for m in N_unmask:
            x[:, m*chunk_step*chunk_size:m*chunk_step*chunk_size+chunk_size, 1] = 0
            y[:, m*chunk_step*chunk_size:m*chunk_step*chunk_size+chunk_size, 1] = 0
        for m in E_unmask:
            x[:, m*chunk_step*chunk_size:m*chunk_step*chunk_size+chunk_size, 2] = 0
            y[:, m*chunk_step*chunk_size:m*chunk_step*chunk_size+chunk_size, 2] = 0

        return F.mse_loss(x, y)

class TS2Vec(nn.Module):
    def __init__(self, d_model, d_ffn, emb_dim, d_out, nhead, n_layers, model_opt, dropout=0.1, inference=False, mask_latent=False):
        super(TS2Vec, self).__init__()

        self.inference = inference
        self.mask_latent = mask_latent

        self.input_projection = nn.Linear(d_model, d_out)
        self.subsample = nn.Sequential(
            nn.Conv1d(d_out, d_out, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(d_out, d_out, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.model_opt = model_opt
        if model_opt == 'transformer':
            self.transformer = nn.TransformerEncoderLayer(d_model=d_out, nhead=nhead, dim_feedforward=d_ffn, batch_first=True)
            self.encoder = nn.TransformerEncoder(self.transformer, num_layers=n_layers)
        elif model_opt == 'conformer': 
            self.encoder = Conformer(num_classes=emb_dim, input_dim=d_out, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=n_layers, subsample=False)

    def forward(self, wave):
        # wave: (batch, 3, 3000)
    
        wave = self.input_projection(wave.permute(0,2,1)).permute(0,2,1)
        wave = self.subsample(wave).permute(0,2,1)

        if self.mask_latent:
            mask = generate_binomial_mask(wave.size(0), wave.size(1), wave.shape[2]).to(wave.device)
            tmp_wave = wave.clone()
            tmp_wave[~mask] = 0
            wave = tmp_wave

        if not self.inference:
            ts_l = wave.size(1)
            temporal_unit = 0
            crop_l = np.random.randint(low=2 ** (temporal_unit + 1), high=ts_l+1)
            crop_left = np.random.randint(ts_l - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
            crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=wave.size(0))

            wave1 = self.take_per_row(wave, crop_offset + crop_eleft, crop_right - crop_eleft)
            wave2 = self.take_per_row(wave, crop_offset + crop_left, crop_eright - crop_left)

            out1 = self.encoder(wave1)
            out2 = self.encoder(wave2)
           
            return out1[:, -crop_l:], out2[:, :crop_l]
        else:
            out = self.encoder(wave)

            return out

    def take_per_row(self, A, indx, num_elem):
        all_indx = indx[:,None] + np.arange(num_elem)
        
        return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

class CPC_model(nn.Module):
    def __init__(self, d_model, emb_dim, d_ffn, model_opt, n_layers, nhead, pred_timesteps=5, num_negatives=40, inference=False):
        super(CPC_model, self).__init__()
        
        self.pred_timesteps = pred_timesteps
        self.emb_dim = emb_dim
        self.d_ffn = d_ffn
        self.num_negatives = num_negatives
        self.model_opt = model_opt 
        self.inference = inference

        self.encoder = nn.Sequential(nn.Conv1d(3, emb_dim, kernel_size=10, stride=3),
                                    nn.ReLU(),
                                    nn.Conv1d(emb_dim, emb_dim, kernel_size=8, stride=2),
                                    nn.ReLU(),
                                    nn.Conv1d(emb_dim, emb_dim, kernel_size=4, stride=1),
                                    nn.ReLU())
        
        if model_opt == 'GRU':
            self.ar_model = nn.GRU(emb_dim, d_ffn, batch_first=True, dropout=0.1)
        elif model_opt == 'transformer':
            self.linear_proj = nn.Linear(emb_dim, d_ffn)
            self.transformer = nn.TransformerEncoderLayer(d_model=d_ffn, nhead=nhead, dim_feedforward=d_ffn*2, batch_first=True)
            self.ar_model = nn.TransformerEncoder(self.transformer, num_layers=n_layers)

        if not inference:
            self.W = nn.ModuleList([nn.Linear(d_ffn, emb_dim) for _ in range(pred_timesteps)])
        else:
            self.W = nn.Linear(d_ffn, emb_dim)
        
    def forward(self, wave):
        batch_size, _, seq_len = wave.size()
        
        if not self.inference:
            # Randomly pick the timestep 't'
            t_samples = torch.randint(492-self.pred_timesteps, size=(1,)).long()
            
            # z -> 經過 encoder 產生的 representation (batch_size, timesteps, emb_dim)
            z = self.encoder(wave).permute(0,2,1)

            # 把要預測的幾個時間點的 encoded 'z' 記錄下來當作 positive
            pos = torch.empty((self.pred_timesteps, batch_size, self.emb_dim)).float()
            for i in np.arange(1, self.pred_timesteps+1):
                pos[i-1] = z[:, t_samples+i, :].view(batch_size, self.emb_dim)
            
            # 將時間點 't' 與之前的所有 timesteps 的 'z' 輸入進 AR model
            forward_seq = z[:, :t_samples+1, :]

            # o -> 經過 AR model 產生的 context (batch_size, timesteps, d_ffn)
            if self.model_opt == 'GRU':
                o, _ = self.ar_model(forward_seq)
            elif self.model_opt == 'transformer':
                forward_seq = self.linear_proj(forward_seq)
                o = self.ar_model(forward_seq)
            
            # AR model 的 output，取最後一個時間點當作是 context 'c'
            c_t = o[:, t_samples, :].view(batch_size, self.d_ffn)

            # 將 context 'c' 去乘上對應的 weight matrix 得到未來幾個時間點的 representation 'z' -> pred
            pred = torch.empty((self.pred_timesteps, batch_size, self.emb_dim)).float()
            for i in np.arange(0, self.pred_timesteps):
                pred[i] = self.W[i](c_t).float()

            # 隨機從 sequence 中 samples 出幾個時間點的 representations 'z' -> neg
            pos_idx = [t_samples+i for i in range(1, self.pred_timesteps+1)]
            neg_idx = np.array(random.sample([i for i in range(492) if i not in pos_idx], self.num_negatives))
            neg = z[:, neg_idx, :].clone().float()

            return pred, pos, neg

        # inference
        else:
            forward_seq = self.encoder(wave).permute(0,2,1)

            if self.model_opt == 'GRU':
                o, _ = self.ar_model(forward_seq)
                o = self.W(o)

            elif self.model_opt == 'transformer':
                forward_seq = self.linear_proj(forward_seq)
                o = self.ar_model(forward_seq)
                o = self.W(o)

            return o
    
class TS_TCC_model(nn.Module):
    def __init__(self, emb_dim, d_ffn, n_layers, nhead, pred_timesteps, inference=False):
        super(TS_TCC_model, self).__init__()

        self.pred_timesteps = pred_timesteps
        self.inference = inference

        self.encoder = TSTCC_encoder()
        self.W = nn.ModuleList([nn.Linear(emb_dim, 128) for _ in range(pred_timesteps)])
        self.lsoftmax = nn.LogSoftmax()
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, 128 // 2),
            nn.BatchNorm1d(128 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(128 // 2, 128 // 4),
        )
        self.seq_transformer = Seq_Transformer(patch_size=128, dim=emb_dim, depth=n_layers, heads=nhead, mlp_dim=d_ffn, inference=inference)

    def forward(self, wave):        
        if not self.inference:
            aug1, aug2 = DataTransform(wave.cpu())
            aug1, aug2 = self.encoder(aug1.to(wave.device)), self.encoder(aug2.to(wave.device))

            # normalize projection feature vectors
            aug1, aug2 = F.normalize(aug1, dim=1), F.normalize(aug2, dim=1)
            aug1, aug2 = aug1.transpose(1, 2), aug2.transpose(1, 2)
            
            # aug1: weak, aug2: strong        
            batch, seq_len, _ = aug1.size()
            
            t_samples = torch.randint(seq_len - self.pred_timesteps, size=(1,)).long().to(aug1.device)  # randomly pick time stamps

            nce = 0  # average over timestep and batch
            encode_samples_w = torch.empty((self.pred_timesteps, batch, 128)).float().to(aug1.device)
            encode_samples_s = torch.empty((self.pred_timesteps, batch, 128)).float().to(aug1.device)
            
            for i in np.arange(1, self.pred_timesteps + 1):
                encode_samples_w[i - 1] = aug2[:, t_samples + i, :].view(batch, 128)
                encode_samples_s[i - 1] = aug1[:, t_samples + i, :].view(batch, 128)
            forward_seq_w = aug1[:, :t_samples + 1, :]
            forward_seq_s = aug2[:, :t_samples + 1, :]
            
            c_t_w = self.seq_transformer(forward_seq_w)
            c_t_s = self.seq_transformer(forward_seq_s)
        
            pred_w = torch.empty((self.pred_timesteps, batch, 128)).float().to(aug1.device)
            pred_s = torch.empty((self.pred_timesteps, batch, 128)).float().to(aug1.device)
            for i in np.arange(0, self.pred_timesteps):
                linear = self.W[i]
                pred_w[i] = linear(c_t_w)
                pred_s[i] = linear(c_t_s)
                
            for i in np.arange(0, self.pred_timesteps):
                total = torch.mm(encode_samples_w[i], torch.transpose(pred_w[i], 0, 1))
                nce += torch.sum(torch.diag(self.lsoftmax(total)))

                total = torch.mm(encode_samples_s[i], torch.transpose(pred_s[i], 0, 1))
                nce += torch.sum(torch.diag(self.lsoftmax(total)))

            nce /= -1. * batch * self.pred_timesteps * 2
            
            return nce, self.projection_head(c_t_w), self.projection_head(c_t_s)
        else:
            z = self.encoder(wave)
            z = F.normalize(z, dim=1).transpose(1, 2)

            z = self.seq_transformer(z)

            return z
# ============== Embedding before encoder & decoder ============== #

# ============== Pretrained Conformer ============== #
class Conformer_TS2Vec(nn.Module):
    def __init__(self, d_model, emb_dim, d_ffn, enc_layers, nhead, inference=False):
        super(Conformer_TS2Vec, self).__init__()

        self.inference = inference

        self.input_projection = nn.Linear(3, d_model)
        self.encoder = Conformer(num_classes=emb_dim, input_dim=d_model, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=enc_layers, only_subsample=True)
        self.conformer = Conformer(num_classes=emb_dim, input_dim=emb_dim, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=enc_layers, subsample=False)

    def forward(self, wave):
        wave = wave.permute(0,2,1)

        # input projection layer
        wave = self.input_projection(wave)

        # timestamp masking
        mask = generate_binomial_mask(wave.size(0), wave.size(1), wave.shape[2]).to(wave.device)
        tmp_wave = wave.clone()
        tmp_wave[~mask] = 0
        wave = tmp_wave

        wave, _ = self.encoder(wave, 3000)

        # encoder
        if not self.inference:
            ts_l = wave.size(1)
            temporal_unit = 0
            crop_l = np.random.randint(low=2 ** (temporal_unit + 1), high=ts_l+1)
            crop_left = np.random.randint(ts_l - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
            crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=wave.size(0))

            wave1 = self.take_per_row(wave, crop_offset + crop_eleft, crop_right - crop_eleft)
            wave2 = self.take_per_row(wave, crop_offset + crop_left, crop_eright - crop_left)

            out1, _ = self.conformer(wave1, 3000)
            out2, _ = self.conformer(wave2, 3000)
            
            return out1[:, -crop_l:], out2[:, :crop_l]
        else:
            out, _ = self.conformer(wave, 3000)

            return out

    def take_per_row(self, A, indx, num_elem):
        all_indx = indx[:,None] + np.arange(num_elem)
        
        return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

class Conformer_CPC(nn.Module):
    def __init__(self, d_model, emb_dim, d_ffn, nhead, enc_layers, pred_timesteps=5, num_negatives=40, inference=False):
        super(Conformer_CPC, self).__init__()

        self.pred_timesteps = pred_timesteps
        self.emb_dim = emb_dim
        self.d_ffn = d_ffn
        self.num_negatives = num_negatives
        self.inference = inference

        self.input_projection = nn.Linear(3, d_model)
        self.encoder = Conformer(num_classes=emb_dim, input_dim=d_model, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=enc_layers, only_subsample=True)
        self.ar_model = Conformer(num_classes=emb_dim, input_dim=emb_dim, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=enc_layers, subsample=False)

        if not inference:
            self.W = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(pred_timesteps)])

    def forward(self, wave):
        batch_size, _, seq_len = wave.size()
        
        wave = self.input_projection(wave.permute(0,2,1))
        
        if not self.inference:
            # Randomly pick the timestep 't'
            t_samples = torch.randint(749-self.pred_timesteps, size=(1,)).long()
            
            # z -> 經過 encoder 產生的 representation (batch_size, timesteps, emb_dim)

            z, _ = self.encoder(wave, 3000)
            
            # 把要預測的幾個時間點的 encoded 'z' 記錄下來當作 positive
            pos = torch.empty((self.pred_timesteps, batch_size, self.emb_dim)).float()
            for i in np.arange(1, self.pred_timesteps+1):
                pos[i-1] = z[:, t_samples+i, :].view(batch_size, self.emb_dim)
            
            # 將時間點 't' 與之前的所有 timesteps 的 'z' 輸入進 AR model
            forward_seq = z[:, :t_samples+1, :]

            # o -> 經過 AR model 產生的 context (batch_size, timesteps, d_ffn)
            o, _ = self.ar_model(forward_seq, 3000)
            
            # AR model 的 output，取最後一個時間點當作是 context 'c'
            c_t = o[:, t_samples, :].view(batch_size, self.emb_dim)

            # 將 context 'c' 去乘上對應的 weight matrix 得到未來幾個時間點的 representation 'z' -> pred
            pred = torch.empty((self.pred_timesteps, batch_size, self.emb_dim)).float()
            for i in np.arange(0, self.pred_timesteps):
                pred[i] = self.W[i](c_t).float()

            # 隨機從 sequence 中 samples 出幾個時間點的 representations 'z' -> neg
            pos_idx = [t_samples+i for i in range(1, self.pred_timesteps+1)]
            neg_idx = np.array(random.sample([i for i in range(749) if i not in pos_idx], self.num_negatives))
            neg = z[:, neg_idx, :].clone().float()

            return pred, pos, neg

        # inference
        else:
            forward_seq, _ = self.encoder(wave, 3000)
            
            o, _ = self.ar_model(forward_seq, 3000)
            
            return o
    
class Conformer_TSTCC(nn.Module):
    def __init__(self, emb_dim, d_ffn, nhead, enc_layers, d_model, pred_timesteps=5, inference=False):
        super(Conformer_TSTCC, self).__init__()

        self.pred_timesteps = pred_timesteps
        self.inference = inference
        self.emb_dim = emb_dim

        self.input_projection = nn.Linear(3, d_model)
        self.encoder = Conformer(num_classes=emb_dim, input_dim=d_model, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=enc_layers, only_subsample=True)
        self.conformer = Conformer(num_classes=emb_dim, input_dim=emb_dim, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=enc_layers, subsample=False)
        self.W = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(pred_timesteps)])
        self.lsoftmax = nn.LogSoftmax()
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, 128 // 2),
            nn.BatchNorm1d(128 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(128 // 2, 128 // 4),
        )    

    def forward(self, wave):
        if not self.inference:
            aug1, aug2 = DataTransform(wave.cpu())
            aug1, aug2 = self.input_projection(aug1.to(wave.device).permute(0,2,1)).permute(0,2,1), self.input_projection(aug2.to(wave.device).permute(0,2,1)).permute(0,2,1)
            
            aug1, _ = self.encoder(aug1.permute(0,2,1), 3000)
            aug2, _ = self.encoder(aug2.permute(0,2,1), 3000)
            # normalize projection feature vectors
            aug1, aug2 = F.normalize(aug1, dim=1), F.normalize(aug2, dim=1)
            # aug1, aug2 = aug1.transpose(1, 2), aug2.transpose(1, 2)
            
            # aug1: weak, aug2: strong        
            batch, seq_len, _ = aug1.size()
            
            t_samples = torch.randint(seq_len - self.pred_timesteps, size=(1,)).long().to(aug1.device)  # randomly pick time stamps

            nce = 0  # average over timestep and batch

            # encode_samples: positive samples
            encode_samples_w = torch.empty((self.pred_timesteps, batch, self.emb_dim)).float().to(aug1.device)
            encode_samples_s = torch.empty((self.pred_timesteps, batch, self.emb_dim)).float().to(aug1.device)
            
            for i in np.arange(1, self.pred_timesteps + 1):
                encode_samples_w[i - 1] = aug2[:, t_samples + i, :].view(batch, self.emb_dim)
                encode_samples_s[i - 1] = aug1[:, t_samples + i, :].view(batch, self.emb_dim)
            forward_seq_w = aug1[:, :t_samples + 1, :]
            forward_seq_s = aug2[:, :t_samples + 1, :]
            
            c_t_w, _ = self.conformer(forward_seq_w, 3000)
            c_t_s, _ = self.conformer(forward_seq_s, 3000)
            c_t_w, c_t_s = c_t_w[:, t_samples, :].view(batch, self.emb_dim), c_t_s[:, t_samples, :].view(batch, self.emb_dim)

            pred_w = torch.empty((self.pred_timesteps, batch, self.emb_dim)).float().to(aug1.device)
            pred_s = torch.empty((self.pred_timesteps, batch, self.emb_dim)).float().to(aug1.device)
            
            for i in np.arange(0, self.pred_timesteps):
                linear = self.W[i]
                pred_w[i] = linear(c_t_w)
                pred_s[i] = linear(c_t_s)
                
            for i in np.arange(0, self.pred_timesteps):
                total = torch.mm(encode_samples_w[i], torch.transpose(pred_w[i], 0, 1))
                nce += torch.sum(torch.diag(self.lsoftmax(total)))
                
                total = torch.mm(encode_samples_s[i], torch.transpose(pred_s[i], 0, 1))
                nce += torch.sum(torch.diag(self.lsoftmax(total)))
                
            nce /= -1. * batch * self.pred_timesteps * 2

            return nce, self.projection_head(c_t_w), self.projection_head(c_t_s)
        else:
            wave = self.input_projection(wave.permute(0,2,1))
            
            z, _ = self.encoder(wave, 3000)
            
            z = F.normalize(z, dim=1)
            
            z, _ = self.conformer(z, 3000)
            
            return z
# ============== Pretrained Conformer ============== #