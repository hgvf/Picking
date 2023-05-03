import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from emb_model import *
from conformer import *
from conformer.encoder import *
from conformer.activation import GLU

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=3000, return_vec=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.return_vec = return_vec

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if not self.return_vec: 
            # x: (batch_size*num_windows, window_size, input_dim)
            x = x[:] + self.pe.squeeze()

            return self.dropout(x)
        else:
            return self.pe.squeeze()

class Permute(nn.Module):
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)

class cross_attn(nn.Module):
    def __init__(self, nhead, d_k, d_v, d_model, dropout=0.1):
        super(cross_attn, self).__init__()
        
        self.nhead = nhead
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, nhead * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, nhead * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, nhead * d_v, bias=False)
        self.fc = nn.Linear(nhead * d_v, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None, mean=None, std=None):
        d_k, d_v, nhead = self.d_k, self.d_v, self.nhead
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, nhead, d_k)
        k = self.w_ks(k).view(sz_b, len_k, nhead, d_k)
        v = self.w_vs(v).view(sz_b, len_v, nhead, d_k)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        
        if mean is not None and std is not None:
            mean = mean.unsqueeze(1).unsqueeze(1)
            std = std.unsqueeze(1).unsqueeze(1)
            attn = torch.matmul(q, k.transpose(-2, -1)) * mean + std
            attn = attn / d_k**0.5
        else:
            attn = torch.matmul(q / d_k**0.5, k.transpose(-2, -1))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        
        output = self.dropout(self.fc(output))
        output += residual
        
        output = self.layer_norm(output)
        
        return output

class cross_attn_layer(nn.Module):
    def __init__(self, nhead, d_k, d_v, d_model, conformer_class, d_ffn):
        super(cross_attn_layer, self).__init__()
        
        self.cross_attn = cross_attn(nhead=nhead, d_k=d_k, d_v=d_v, d_model=d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn),
                                    nn.ReLU(),
                                    nn.Linear(d_ffn, d_model),
                                )   
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        
        # d_model: dimension of query vector
        self.proj = False
        if d_model != conformer_class:
            self.proj = True
            self.projector = nn.Conv1d(d_model, conformer_class, kernel_size=3, padding='same')
            
    def forward(self, q, k, v, mean=None, std=None):
        out_attn = self.cross_attn(q, k, v, mean=mean, std=std)
            
        out = self.layer_norm(self.ffn(out_attn) + out_attn)
        out = self.dropout(out)
        
        if self.proj:
            out = self.projector(out.permute(0,2,1)).permute(0,2,1)
        return out

class cross_attn_AWG(nn.Module):
    def __init__(self, nhead, d_k, d_v, d_model, norm_type, l, dropout=0.1):
        super(cross_attn_AWG, self).__init__()
        
        self.nhead = nhead
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, nhead * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, nhead * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, nhead * d_v, bias=False)
        self.fc = nn.Linear(nhead * d_v, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_type = norm_type
        self.l = l
        
    def forward(self, q, k, v, mask=None):
        d_k, d_v, nhead = self.d_k, self.d_v, self.nhead
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, nhead, d_k)
        k = self.w_ks(k).view(sz_b, len_k, nhead, d_k)
        v = self.w_vs(v).view(sz_b, len_v, nhead, d_k)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
            
        attn = torch.matmul(q / d_k**0.5, k.transpose(-2, -1))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        # AWG module
        if self.norm_type == 'mean':
            # mean as threshold
            mean_W = torch.mean(attn, dim=-1)
            AWG_mask = torch.where(attn >= mean_W[:, :, :, None], 1, 0)
        else:
            # l-largest as threshold
            l = self.l
            candidate = torch.topk(attn, l, dim=-1)
            threshold = candidate[0][:, :, :, -1]
            AWG_mask = torch.where(attn >= threshold[:, :, :, None], 1, 0)

        W = attn * AWG_mask
        W_out = W / torch.sum(W, dim=-1)[:, :, :, None]
        output = torch.matmul(W_out, v)
        
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        
        output = self.dropout(self.fc(output))
        output += residual
        
        output = self.layer_norm(output)
        
        return output

class MGAN(nn.Module):
    def __init__(self, nhead, d_k, d_v, d_model, conformer_class, d_ffn, norm_type, l):
        super(MGAN, self).__init__()

        self.layer_norm = nn.LayerNorm(conformer_class, eps=1e-6)
        self.fc1 = nn.Linear(conformer_class, conformer_class)
        self.fc2 = nn.Linear(conformer_class, conformer_class)
        self.sigmoid = nn.Sigmoid()
        self.glu = GLU(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.cross_attn_AWG = cross_attn_AWG(nhead=nhead, d_k=d_k, d_v=d_v, d_model=conformer_class, norm_type=norm_type, l=l)

    def forward(self, q, k, v):
        # pre-layernorm
        q = self.layer_norm(q)

        # SG module
        sg = self.sigmoid(self.fc1(q)) * self.fc2(q)

        awg = self.cross_attn_AWG(sg, k, v)

        out_attn = self.glu(torch.cat((sg, awg), dim=-1))

        return self.dropout(q + out_attn)

class Emb(nn.Module):
    def __init__(self, n_class, emb_dim):
        super(Emb, self).__init__()
        
        assert emb_dim // 2 >= n_class, "emb_dim // 2 must larger or equal than n_class"
        
        self.conv = nn.Sequential(nn.Conv2d(1, 3, (3,3), padding='same'),
                                 nn.BatchNorm2d(3),
                                 nn.ReLU(),
                                 nn.Conv2d(3, 8, (3,3), padding='same'),
                                 nn.BatchNorm2d(8),
                                 nn.ReLU(),
                                 nn.Conv2d(8, 16, (3,3), padding='same'),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU())
        
        self.lstm = nn.LSTM(48, 64, bidirectional=True, batch_first=True, dropout=0.1, num_layers=2)
        
        self.fc = nn.Linear(n_class, emb_dim//2)
        
        self.embedding = nn.Parameter(torch.rand(emb_dim // 2, emb_dim))
        
    def forward(self, x):        
        x = self.conv(x)
        batch, channel, seq_len, dim = x.size()
        
        x = x.contiguous().view(batch, seq_len, channel*dim)
        x, _ = self.lstm(x)
        x = self.fc(x)
        
        # ids: (batch, seq_len)
        ids = torch.argmax(x, dim=-1)
        
        emb = self.embedding[ids]
        
        return emb

class ScalarProjector(nn.Module):
    def __init__(self, output_dim, enc_dim=3, seq_len=3000):
        super(ScalarProjector, self).__init__()

        self.conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        self.mlp = nn.Sequential(nn.Linear(2*enc_dim, 16),
                                nn.ReLU(),
                                nn.Linear(16, 32),
                                nn.ReLU(),)
        self.out = nn.Linear(32, output_dim)

    def forward(self, wf, stats):
        # wf must follow in shape: (batch, seq_len, chn)
        out = self.conv(wf.permute(0,2,1))

        # stats: (batch, 3)
        stats = stats.unsqueeze(1)
        out = torch.cat([out, stats], dim=1)
        out = out.view(wf.shape[0], -1)

        out = self.out(self.mlp(out))

        return out

class Residual_Unet(nn.Module):
    def __init__(self, conformer_class, nhead, d_ffn ,dec_layers, res_dec):
        super(Residual_Unet, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv1d(conformer_class, conformer_class*2, kernel_size=3, stride=2),
                                       nn.BatchNorm1d(conformer_class*2),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),)
        
        self.conv2 = nn.Sequential(nn.Conv1d(conformer_class*2, conformer_class*3, kernel_size=5, stride=3),
                                       nn.BatchNorm1d(conformer_class*3),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),)
        
        self.conv3 = nn.Sequential(nn.Conv1d(conformer_class*3, conformer_class*4, kernel_size=7, stride=3),
                                       nn.BatchNorm1d(conformer_class*4),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),)
        
        self.conv4 = nn.Sequential(nn.Conv1d(conformer_class*4, conformer_class*5, kernel_size=9, stride=3),
                                       nn.BatchNorm1d(conformer_class*5),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),)
        
        self.res_dec = res_dec
        if res_dec:
            self.dec_layers = dec_layers
            self.bottleneck = nn.ModuleList([])
            for _ in range(dec_layers):
                bottleneck_layer = nn.TransformerEncoderLayer(d_model=conformer_class*5, nhead=nhead, dim_feedforward=d_ffn)
                self.bottleneck.append(nn.TransformerEncoder(bottleneck_layer, num_layers=1))
        else:
            self.bottleneck_layer = nn.TransformerEncoderLayer(d_model=conformer_class*5, nhead=nhead, dim_feedforward=d_ffn)
            self.bottleneck = nn.TransformerEncoder(self.bottleneck_layer, num_layers=dec_layers)

        self.up1 = nn.Sequential(nn.Upsample(165),
                                      nn.Conv1d(conformer_class*5, conformer_class*4, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(conformer_class*4),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),)
        
        self.up2 = nn.Sequential(nn.Upsample(499),
                                      nn.Conv1d(conformer_class*4, conformer_class*3, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(conformer_class*3),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),)
        
        self.up3 = nn.Sequential(nn.Upsample(1499),
                                      nn.Conv1d(conformer_class*3, conformer_class*2, kernel_size=5, padding='same'),
                                      nn.BatchNorm1d(conformer_class*2),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),)
        
        self.up4 = nn.Sequential(nn.Upsample(3000),
                                      nn.Conv1d(conformer_class*2, conformer_class*1, kernel_size=7, padding='same'),
                                      nn.BatchNorm1d(conformer_class),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),)
    def forward(self, x):
        # print('x: ', x.shape)
        down1_out = self.conv1(x)
        # print('conv1: ', down1_out.shape)
        
        down2_out = self.conv2(down1_out)
        # print('conv2: ', down2_out.shape)
        
        down3_out = self.conv3(down2_out)
        # print('conv3: ', down3_out.shape)
        
        down4_out = self.conv4(down3_out)
        # print('conv4: ', down4_out.shape)
        
        if not self.res_dec:
            bottleneck_out = self.bottleneck(down4_out.permute(0,2,1)).permute(0,2,1)
        else:
            for i, layer in enumerate(self.bottleneck):
                if i == 0:
                    bottleneck_out = layer(down4_out.permute(0,2,1))
                else:
                    bottleneck_out = bottleneck_out + layer(bottleneck_out)
            bottleneck_out = bottleneck_out.permute(0,2,1)
        
        up1_out = self.up1(down4_out)
        # print('up1: ', up1_out.shape)
        
        up2_out = self.up2(up1_out+down3_out)
        # print('up2: ', up2_out.shape)
        
        up3_out = self.up3(up2_out+down2_out)
        # print('up3: ', up3_out.shape)
        
        up4_out = self.up4(up3_out+down1_out)
        # print('up4: ', up4_out.shape)
        
        return up4_out
        
class SingleP_transformer_window(nn.Module):
    def __init__(self, d_ffn, n_head, enc_layers, window_size, dropout=0.1):
        super(SingleP_transformer_window, self).__init__()

        self.pos_emb = PositionalEncoding(12, 0.1, window_size)

        self.transformer = nn.TransformerEncoderLayer(d_model=12, nhead=n_head, dim_feedforward=d_ffn, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=enc_layers)
        
        self.fc = nn.Linear(12, 1)
        self.score = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        nn.init.xavier_uniform_(self.fc.weight)
        self._init_transformer_weight()
        
    def _init_transformer_weight(self):
        nn.init.xavier_uniform_(self.transformer.linear1.weight)
        nn.init.xavier_uniform_(self.transformer.linear2.weight)        
        
    def forward(self, x):
        batch_size, window, channel, wavelen = x.size(0), x.size(1), x.size(2), x.size(3)

        x = x.reshape(-1, channel, wavelen).permute(0, 2, 1)

        x = self.pos_emb(x)
        
        x = self.encoder(x)

        scores = self.score(self.fc(x).squeeze())

        return scores

class SingleP_Conformer(nn.Module):
    def __init__(self, conformer_class, d_model, d_ffn, n_head, enc_layers, dec_layers, norm_type, l, rep_KV, decoder_type='crossattn', encoder_type='conformer', query_type='pos_emb', intensity_MT=False, label_type='p'):
        super(SingleP_Conformer, self).__init__()

        assert encoder_type in ['conformer', 'transformer'], "encoder_type must be one of ['conformer', 'transformer']"
        assert decoder_type in ['upsample', 'crossattn', 'unet', 'MGAN'], "encoder_type must be one of ['upsample', 'crossattn', 'unet', 'MGAN']"

        # =========================================== #
        #                   Encoder                   #
        # =========================================== #
        self.encoder_type = encoder_type
        if encoder_type == 'conformer':
            self.conformer = Conformer(num_classes=conformer_class, input_dim=d_model, encoder_dim=d_ffn, num_attention_heads=n_head, num_encoder_layers=enc_layers)
        elif encoder_type == 'transformer':
            self.subsample = nn.Sequential(
                nn.Conv2d(1, d_ffn, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(d_ffn, d_ffn, kernel_size=3, stride=2),
                nn.ReLU(),
            )
            self.transformer = nn.TransformerEncoderLayer(d_model=d_ffn*2, nhead=n_head, dim_feedforward=d_ffn*4, batch_first=True)
            self.encoder = nn.TransformerEncoder(self.transformer, num_layers=enc_layers)
            conformer_class = d_ffn*2

        self.label_type = label_type
        if label_type == 'p':
            self.fc = nn.Linear(conformer_class, 1)
            self.sigmoid = nn.Sigmoid()
        elif label_type == 'other':
            self.fc = nn.Linear(conformer_class, 2)
            self.softmax = nn.Softmax(dim=-1)
        
        # =========================================== #
        #                   Decoder                   #
        # =========================================== #
        self.query_type = query_type
        self.decoder_type = decoder_type
        self.dec_layers = dec_layers
        self.intensity_MT = intensity_MT
        self.rep_KV = rep_KV
        wave_length = 3000

        if intensity_MT:
            # self.MT_projector = nn.Linear(conformer_class, 7)
            self.MT_projector = nn.Sequential(nn.Conv1d(conformer_class, 24, kernel_size=7, stride=3),
                                nn.BatchNorm1d(24),
                                nn.ReLU(),
                                nn.Conv1d(24, 32, kernel_size=5, stride=3),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Conv1d(32, 36, kernel_size=3, stride=2),
                                nn.BatchNorm1d(36),
                                nn.ReLU(),
                                nn.Flatten(),
                                nn.Linear(1440, 3),
                                )

        if self.query_type == 'pos_emb':
                self.pos_emb = PositionalEncoding(conformer_class, max_len=wave_length, return_vec=True)
        elif self.query_type == 'stft':
                # number of frequency components
                dim_stft = 64
                self.crossAttn_stft_posEmb = cross_attn_layer(n_head, dim_stft//n_head, dim_stft//n_head, dim_stft, conformer_class, d_ffn)
                self.stft_pos_emb = PositionalEncoding(dim_stft, max_len=wave_length, return_vec=True)

        if decoder_type == 'upsample':
            self.upconv = nn.Sequential(nn.Upsample(1500),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(2000),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(2500),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(wave_length),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU())        
        elif decoder_type == 'crossattn':
            self.crossAttnLayer = nn.ModuleList([cross_attn_layer(n_head, conformer_class//n_head, conformer_class//n_head, conformer_class, conformer_class, d_ffn)
                                                for _ in range(dec_layers)]
                                                )
        elif decoder_type == 'MGAN':
            self.crossAttnLayer = cross_attn_layer(n_head, conformer_class//n_head, conformer_class//n_head, conformer_class, conformer_class, d_ffn)
            self.MGANs = nn.ModuleList([MGAN(n_head, conformer_class//n_head, conformer_class//n_head, conformer_class, conformer_class, d_ffn, norm_type, l)
                                        for _ in range(dec_layers)])
            
        elif decoder_type == 'unet':
            self.decoder = Conformer(input_dim=conformer_class, encoder_dim=d_ffn, num_attention_heads=n_head, subsample=False, num_encoder_layers=dec_layers, num_classes=d_ffn)
            self.upconv = nn.Sequential(
                nn.Upsample((5, 1499), mode='bilinear', align_corners=True),
                nn.Conv2d(d_ffn, d_ffn, 3, padding='same'),
                nn.ReLU(),
                nn.Upsample((12, wave_length), mode='bilinear', align_corners=True),
                nn.Conv2d(d_ffn, 1, 3, padding='same'),
                nn.ReLU(),
            )
            self.prediction_head = nn.Linear(d_model, 1)
        
    def forward(self, wave, input_lengths=3000, stft=None):
        wave = wave.permute(0,2,1)
        
        if self.encoder_type == 'conformer':
            out, _ = self.conformer(wave, input_lengths)

            if self.intensity_MT:
                MT_out = self.MT_projector(out.permute(0,2,1))
                MT_out = F.softmax(MT_out, dim=-1)

            # adding noise to encoded representation
            #out = out + torch.randn(out.shape).to(out.device)
        elif self.encoder_type == 'transformer':
            wave = self.subsample(wave.permute(0,2,1).unsqueeze(1))
            batch_size, channels, sumsampled_dim, subsampled_lengths = wave.size()

            wave = wave.permute(0, 3, 1, 2)
            wave = wave.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)

            out = self.encoder(wave)

        if self.decoder_type == 'upsample':
            out = self.upconv(out.permute(0,2,1))
            out = self.sigmoid(self.fc(out.permute(0,2,1)))
        elif self.decoder_type == 'crossattn':
            for i in range(self.dec_layers):
                if i == 0:
                    if self.query_type == 'pos_emb':
                        pos_emb = self.pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
                        dec_out = self.crossAttnLayer[i](pos_emb, out, out)
                    elif self.query_type == 'stft':
                        stft_pos_emb = self.stft_pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
                        stft_q = self.crossAttn_stft_posEmb(stft_pos_emb, stft, stft)
                        
                        dec_out = self.crossAttnLayer[i](stft_q, out, out)
                        
                else:
                    if self.rep_KV:
                        dec_out = self.crossAttnLayer[i](dec_out, out, out)
                    else:
                        dec_out = self.crossAttnLayer[i](dec_out, dec_out, dec_out)
                
        elif self.decoder_type == 'MGAN':
            # cross-attention
            
            if self.query_type == 'pos_emb':
                pos_emb = self.pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
                crossattn_out = self.crossAttnLayer(pos_emb, out, out)
            
            elif self.query_type == 'stft':
                stft_pos_emb = self.stft_pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
                stft_q = self.crossAttn_stft_posEmb(stft_pos_emb, stft, stft)
                crossattn_out = self.crossAttnLayer(stft_q, out, out)

            # MGANs
            for i in range(self.dec_layers):
                if self.rep_KV:
                    dec_out = self.MGANs[i](crossattn_out, out, out)
                else:
                    dec_out = self.MGANs[i](crossattn_out, crossattn_out, crossattn_out)

        if self.label_type == 'p':
            out = self.sigmoid(self.fc(dec_out))
        elif self.label_type == 'other':
            out = self.softmax(self.fc(dec_out))

        if not self.intensity_MT:
            return out
        else:
            return out, MT_out

    def _freeze_weight(self):
        for param in self.conformer.parameters():
            param.requires_grad = False

        # for param in self.fc.parameters():
        #     param.requires_grad = False

        # for param in self.crossAttnLayer.parameters():
        #     param.requires_grad = False

class SingleP_Conformer_spectrogram(nn.Module):
    def __init__(self, conformer_class, d_model, d_ffn, n_head, enc_layers, dec_layers, upsample=False, learnable_query=False, subsample=False, dim_spectrogram='1D'):
        super(SingleP_Conformer_spectrogram, self).__init__()

        self.conformer = Conformer(subsample=subsample, num_classes=conformer_class, input_dim=d_model, encoder_dim=d_ffn, num_attention_heads=n_head, num_encoder_layers=enc_layers)

        self.fc = nn.Linear(conformer_class, 1)
        self.sigmoid = nn.Sigmoid()

        self.dec_layers = dec_layers
        self.upsample = upsample
        self.learnable_query = learnable_query
        if upsample:
            self.upconv = nn.Sequential(nn.Upsample(1500),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(2000),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(2500),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(3000),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU())        
        else:
            self.crossAttnLayer = nn.ModuleList([cross_attn_layer(n_head, conformer_class//n_head, conformer_class//n_head, d_model, conformer_class, d_ffn)
                                                for _ in range(dec_layers)]
                                                )

            self.pos_emb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)

        self.dim_spectrogram = dim_spectrogram
        if self.dim_spectrogram == '2D':    
            patch_height = 8
            patch_width = 5
            patch_dim = 3 * patch_height * patch_width
            dim = d_model

            self.patch = nn.Sequential(
                        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                        nn.Linear(patch_dim, dim),
                    )
    
    def forward(self, wave, input_lengths=56):
        # print('ori: ', wave.shape)
        if self.dim_spectrogram == '2D':
            wave = self.patch(wave)
        # print('wave: ', wave.shape)
        out, _ = self.conformer(wave, input_lengths)
        
        if self.upsample:
            out = self.upconv(out.permute(0,2,1))
            out = self.sigmoid(self.fc(out.permute(0,2,1)))
        else:
            pos_emb = self.pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
                
            for i in range(self.dec_layers):
                if i == 0:
                    dec_out = self.crossAttnLayer[i](pos_emb, out, out)
                else:
                    dec_out = self.crossAttnLayer[i](dec_out, out, out)
                
            out = self.sigmoid(self.fc(dec_out))
        
        return out

class SingleP_WaveformEmb(nn.Module):
    def __init__(self, conformer_class, d_model, d_ffn, n_head, enc_layers, dec_layers, n_class, 
                emb_dim, emb_type, emb_d_ffn, emb_layers, pretrained_emb=None, decoder_type='crossattn', encoder_type='conformer', emb_model_opt=None):
        super(SingleP_WaveformEmb, self).__init__()

        # pretrained embedding layer
        self.emb_type = emb_type
        if emb_type == 'transformer_MLM':
            self.emb = MLM(0, 0, 0, 'transformer', 
                        False, True, *(d_model, n_head, emb_d_ffn, emb_layers, emb_dim))
        elif emb_type == 'transformer':
            self.emb = TransformerEmb(d_model=d_model, d_ffn=emb_d_ffn, d_out=emb_dim, nhead=n_head, n_layers=emb_layers)
        elif emb_type == 'TS2Vec':
            self.emb = TS2Vec(d_model=d_model, d_ffn=emb_d_ffn, d_out=emb_dim, nhead=n_head, n_layers=emb_layers, model_opt=emb_model_opt, emb_dim=emb_dim, inference=True)
        elif emb_type == 'linear':
            self.emb = nn.Sequential(Permute(),
                                    nn.Linear(d_model, emb_dim),
                                    PositionalEncoding(d_model=emb_dim))
        elif emb_type == 'lstm':
            self.emb = nn.Sequential(Permute(),
                                    nn.LSTM(d_model, emb_dim//2, batch_first=True, dropout=0.1, num_layers=enc_layers, bidirectional=True))

        elif emb_type == 'CPC':
            self.emb = model = CPC_model(d_model=d_model, emb_dim=emb_dim, d_ffn=emb_d_ffn, model_opt=emb_model_opt, n_layers=emb_layers, nhead=n_head, inference=True)

        if pretrained_emb is not None:
            pretrained_emb = 'results_emb/'+pretrained_emb+'/model.pt'
            checkpoint = torch.load(pretrained_emb)
            print('loading pretrained embedding: ', pretrained_emb)
            self.emb.load_state_dict(checkpoint['model'], strict=False)
        else:
            print('embedding train from scratch...')
            
        # downstream model
        self.encoder_type = encoder_type
        if encoder_type == 'conformer':
            if emb_type == 'CPC':
                self.conformer = Conformer(num_classes=conformer_class, input_dim=emb_dim, encoder_dim=d_ffn, num_attention_heads=n_head, num_encoder_layers=enc_layers, subsample=False)
            else:
                self.conformer = Conformer(num_classes=conformer_class, input_dim=emb_dim, encoder_dim=d_ffn, num_attention_heads=n_head, num_encoder_layers=enc_layers)
        elif encoder_type == 'transformer':
            self.subsample = nn.Sequential(
                nn.Conv2d(1, d_ffn, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(d_ffn, d_ffn, kernel_size=3, stride=2),
                nn.ReLU(),
            )
            self.transformer = nn.TransformerEncoderLayer(d_model=d_ffn*2, nhead=n_head, dim_feedforward=d_ffn*4, batch_first=True)
            self.encoder = nn.TransformerEncoder(self.transformer, num_layers=enc_layers)
            conformer_class = d_ffn*2

        self.fc = nn.Linear(conformer_class, 1)
        self.sigmoid = nn.Sigmoid()

        self.decoder_type = decoder_type
        self.dec_layers = dec_layers
        if decoder_type == 'upsample':
            self.upconv = nn.Sequential(nn.Upsample(1500),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(2000),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(2500),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(3000),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU())        
        elif decoder_type == 'crossattn':
            self.crossAttnLayer = nn.ModuleList([cross_attn_layer(n_head, conformer_class//n_head, conformer_class//n_head, d_model, conformer_class, d_ffn)
                                                for _ in range(dec_layers)]
                                                )
            self.pos_emb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)
        elif decoder_type == 'MGAN':
            self.pos_emb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)
            self.crossAttnLayer = cross_attn_layer(n_head, conformer_class//n_head, conformer_class//n_head, d_model, conformer_class, d_ffn)
            self.MGANs = nn.ModuleList([MGAN(n_head, conformer_class//n_head, conformer_class//n_head, d_model, conformer_class, d_ffn, norm_type, l)
                                        for _ in range(dec_layers)])                           
        elif decoder_type == 'unet':
            self.decoder = Conformer(input_dim=conformer_class, encoder_dim=d_ffn, num_attention_heads=n_head, subsample=False, num_encoder_layers=dec_layers, num_classes=d_ffn)
            self.upconv = nn.Sequential(
                nn.Upsample((5, 1499), mode='bilinear', align_corners=True),
                nn.Conv2d(d_ffn, d_ffn, 3, padding='same'),
                nn.ReLU(),
                nn.Upsample((12, 3000), mode='bilinear', align_corners=True),
                nn.Conv2d(d_ffn, 1, 3, padding='same'),
                nn.ReLU(),
            )
            self.prediction_head = nn.Linear(d_model, 1)

    def forward(self, x):
        if self.emb_type == 'lstm':
            wave, _ = self.emb(x)
        else:
            wave = self.emb(x)

        if self.encoder_type == 'conformer':
            out, _ = self.conformer(wave, 3000)
        elif self.encoder_type == 'transformer':
            wave = self.subsample(wave.permute(0,2,1).unsqueeze(1))
            batch_size, channels, sumsampled_dim, subsampled_lengths = wave.size()

            wave = wave.permute(0, 3, 1, 2)
            wave = wave.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)

            out = self.encoder(wave)

        if self.decoder_type == 'upsample':
            out = self.upconv(out.permute(0,2,1))
            out = self.sigmoid(self.fc(out.permute(0,2,1)))
        elif self.decoder_type == 'crossattn':
            pos_emb = self.pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
            
            for i in range(self.dec_layers):
                if i == 0:
                    dec_out = self.crossAttnLayer[i](pos_emb, out, out)
                else:
                    dec_out = self.crossAttnLayer[i](dec_out, out, out)
                
            out = self.sigmoid(self.fc(dec_out))
        elif self.decoder_type == 'MGAN':
            pos_emb = self.pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)

            # cross-attention
            crossattn_out = self.crossAttnLayer(pos_emb, out, out)

            # MGANs
            for i in range(self.dec_layers):
                dec_out = self.MGANs[i](crossattn_out)

            out = self.sigmoid(self.fc(dec_out))
        elif self.decoder_type == 'unet':
            dec_out, _ = self.decoder(out, out.shape[1])
            dec_out = self.upconv(dec_out.unsqueeze(1).permute(0, 3, 1, 2))
            dec_out = dec_out[:, 0].permute(0, 2, 1)
            out = self.prediction_head(dec_out)

        return out

class SSL_Conformer(nn.Module):
    def __init__(self, conformer_class, d_model, d_ffn, n_head, enc_layers, dec_layers, norm_type, l, 
                emb_dim, emb_type, emb_d_ffn, emb_layers, pretrained_emb=None, decoder_type='crossattn'):
        super(SSL_Conformer, self).__init__()

        # self-supervised pretraining Conformer
        if emb_type == 'CPC':
            self.encoder = Conformer_CPC(d_model=d_model, emb_dim=conformer_class, d_ffn=emb_d_ffn, enc_layers=emb_layers, nhead=n_head, inference=True)
        elif emb_type == 'TS2Vec':
            self.encoder = Conformer_TS2Vec(d_model=d_model, d_ffn=emb_d_ffn, nhead=n_head, enc_layers=emb_layers, emb_dim=conformer_class, inference=True)
        elif emb_type == 'TSTCC':
            self.encoder = Conformer_TSTCC(emb_dim=conformer_class, d_ffn=emb_d_ffn, enc_layers=emb_layers, nhead=n_head, d_model=d_model, inference=True)  

        # decoder
        if decoder_type == 'upsample':
            self.upconv = nn.Sequential(nn.Upsample(1500),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(2000),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(2500),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(3000),
                                        nn.Conv1d(conformer_class, conformer_class, 11, padding='same'),
                                        nn.ReLU())        
        elif decoder_type == 'crossattn':
            self.crossAttnLayer = nn.ModuleList([cross_attn_layer(n_head, conformer_class//n_head, conformer_class//n_head, d_model, conformer_class, d_ffn)
                                                for _ in range(dec_layers)]
                                                )
            self.pos_emb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)
        elif decoder_type == 'MGAN':
            self.pos_emb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)
            self.crossAttnLayer = cross_attn_layer(n_head, conformer_class//n_head, conformer_class//n_head, d_model, conformer_class, d_ffn)
            self.MGANs = nn.ModuleList([MGAN(n_head, conformer_class//n_head, conformer_class//n_head, d_model, conformer_class, d_ffn, norm_type, l)
                                        for _ in range(dec_layers)])                           

        # other modules
        self.fc = nn.Linear(conformer_class, 1)
        self.sigmoid = nn.Sigmoid()

        self.decoder_type = decoder_type
        self.dec_layers = dec_layers        
        
        if pretrained_emb is not None:
            pretrained_emb = 'results_emb/'+pretrained_emb+'/model.pt'
            checkpoint = torch.load(pretrained_emb, map_location=self.fc.weight.device)
            print('loading pretrained embedding: ', pretrained_emb)
            self.encoder.load_state_dict(checkpoint['model'], strict=False)
        else:
            print('embedding train from scratch...')      

    def forward(self, wave):
        out = self.encoder(wave)

        if self.decoder_type == 'upsample':
            out = self.upconv(out.permute(0,2,1))
            out = self.sigmoid(self.fc(out.permute(0,2,1)))
        elif self.decoder_type == 'crossattn':
            pos_emb = self.pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
            
            for i in range(self.dec_layers):
                if i == 0:
                    dec_out = self.crossAttnLayer[i](pos_emb, out, out)
                else:
                    dec_out = self.crossAttnLayer[i](dec_out, out, out)
                
            out = self.sigmoid(self.fc(dec_out))
        elif self.decoder_type == 'MGAN':
            pos_emb = self.pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)

            # cross-attention
            crossattn_out = self.crossAttnLayer(pos_emb, out, out)

            # MGANs
            for i in range(self.dec_layers):
                dec_out = self.MGANs[i](crossattn_out)

            out = self.sigmoid(self.fc(dec_out))

        return out

class AntiCopy_Conformer(nn.Module):
    def __init__(self, conformer_class, d_model, d_ffn, n_head, enc_layers, dec_layers, norm_type, l, decoder_type='crossattn', encoder_type='conformer'):
        super(AntiCopy_Conformer, self).__init__()

        self.encoder_type = encoder_type
        if encoder_type == 'conformer':
            self.conformer = Conformer(num_classes=conformer_class, input_dim=d_model, encoder_dim=d_ffn, num_attention_heads=n_head, num_encoder_layers=enc_layers)
        
        self.fc = nn.Linear(conformer_class, 3)
        self.softmax = nn.Softmax(dim=-1)

        self.decoder_type = decoder_type
        self.dec_layers = dec_layers
        if decoder_type == 'crossattn':
            self.crossAttnLayer = nn.ModuleList([cross_attn_layer(n_head, conformer_class//n_head, conformer_class//n_head, d_model, conformer_class, d_ffn)
                                                for _ in range(dec_layers)]
                                                )
            self.pos_emb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)
        elif decoder_type == 'MGAN':
            self.pos_emb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)
            self.crossAttnLayer = cross_attn_layer(n_head, conformer_class//n_head, conformer_class//n_head, d_model, conformer_class, d_ffn)
            self.MGANs = nn.ModuleList([MGAN(n_head, conformer_class//n_head, conformer_class//n_head, d_model, conformer_class, d_ffn, norm_type, l)
                                        for _ in range(dec_layers)])

    def forward(self, wave, input_lengths=3000):
        wave = wave.permute(0,2,1)

        if self.encoder_type == 'conformer':
            out, _ = self.conformer(wave, input_lengths)
        
        if self.decoder_type == 'crossattn':
            pos_emb = self.pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
            
            for i in range(self.dec_layers):
                if i == 0:
                    dec_out = self.crossAttnLayer[i](pos_emb, out, out)
                else:
                    dec_out = self.crossAttnLayer[i](dec_out, out, out)
                
            out = self.softmax(self.fc(dec_out))
        elif self.decoder_type == 'MGAN':
            pos_emb = self.pos_emb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)

            # cross-attention
            crossattn_out = self.crossAttnLayer(pos_emb, out, out)

            # MGANs
            for i in range(self.dec_layers):
                dec_out = self.MGANs[i](crossattn_out)

            out = self.softmax(self.fc(dec_out))
        
        return out

class GRADUATE(nn.Module):
    def __init__(self, conformer_class, d_ffn, nhead, d_model, enc_layers, dec_layers, norm_type, l, 
                 cross_attn_type, seg_proj_type='crossattn', encoder_type='conformer', decoder_type='crossattn', 
                 rep_KV=True, label_type='p', recover_type="crossattn", res_dec=False):
        super(GRADUATE, self).__init__()
        
        dim_stft = 32
        
        # =========================================== #
        #                   Encoder                   #
        # =========================================== #
        # encoded representation 會當作 decoder's Key, Value
        self.rep_KV = rep_KV
        self.seg_proj_type = seg_proj_type
        self.encoder_type = encoder_type
        
        # down-sample layer
        if encoder_type == 'unet_conformer':
            self.down_conv = nn.Sequential(nn.Conv1d(d_model, d_model*3, kernel_size=3, stride=2),
                                           nn.BatchNorm1d(d_model*3),
                                           nn.ReLU(),
                                           nn.Conv1d(d_model*3, d_model*5, kernel_size=5, stride=3),
                                           nn.BatchNorm1d(d_model*5),
                                           nn.ReLU(),
                                           nn.Conv1d(d_model*5, d_model*8, kernel_size=5, stride=3),
                                           nn.BatchNorm1d(d_model*8),
                                           nn.ReLU(),)
            self.conformer = Conformer(subsample=False, num_classes=conformer_class, input_dim=d_model*8, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=enc_layers)
        else:
            self.conformer = Conformer(num_classes=conformer_class, input_dim=d_model, encoder_dim=d_ffn, num_attention_heads=nhead, num_encoder_layers=enc_layers)
        
        if seg_proj_type == 'crossattn':
            self.seg_posEmb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)
            self.seg_crossattn = cross_attn(nhead=nhead, d_k=conformer_class//nhead, d_v=conformer_class//nhead, d_model=conformer_class)
            self.seg_projector = nn.Linear(conformer_class, 1)
        elif seg_proj_type == 'upsample':
            self.upconv = nn.Sequential(nn.Upsample(1500),
                                        nn.Conv1d(conformer_class, conformer_class, 3, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(2000),
                                        nn.Conv1d(conformer_class, conformer_class, 3, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(2500),
                                        nn.Conv1d(conformer_class, conformer_class, 5, padding='same'),
                                        nn.ReLU(),
                                        nn.Upsample(3000),
                                        nn.Conv1d(conformer_class, conformer_class, 5, padding='same'),
                                        nn.ReLU(),)
            self.seg_projector = nn.Sequential(nn.Linear(conformer_class, conformer_class*3),
                                               nn.ReLU(),
                                               nn.Linear(conformer_class*3, conformer_class)) 
        self.sigmoid = nn.Sigmoid()
        
        # =========================================== #
        #               Cross-Attention               #
        # =========================================== #
        '''
        cross_attn_type 
        1) 先將 stft & encoded representation project 到相同長度，再去關注個時間點的頻率能量，最後再把 hidden state 
            復原成原始資料長度做 decode。
                cross_attn(cross_attn(stft, pos_emb), encoded_representation)  => output: (batch, 749, conformer_class)
                
        2) 分別將 encoded representation & stft 各自復原成原始長度後，再以原始長度大小去關注個時間點的頻率能量，最後再 decode。
                cross_attn(cross_attn(stft, pos_emb), cross_attn(encoded_representation, pos_emb)) => output: (batch, 3000, conformer_class)
                
        3) Encoded representation 先經過原始方法 decode 完之後，再與原始長度的 stft 做 cross-attention，以此關注個時間點頻率能量。
                cross_attn(decoder(cross_attn(encoded_representation, pos_emb)), cross_attn(stft, pos_emb)) => output: (batch, 3000, conformer_class)
        '''
        self.cross_attn_type = cross_attn_type
        if cross_attn_type == 1:
            self.stft_posEmb = PositionalEncoding(dim_stft, max_len=749, return_vec=True)
            self.stft_rep_posEmb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)
            self.stft_pos_emb = cross_attn_layer(nhead, dim_stft//nhead, dim_stft//nhead, dim_stft, conformer_class, d_ffn)
            self.stft_rep = cross_attn_layer(nhead, conformer_class//nhead, conformer_class//nhead, conformer_class, conformer_class, d_ffn)
            self.crossattn = cross_attn_layer(nhead, conformer_class//nhead, conformer_class//nhead, conformer_class, conformer_class, d_ffn)
            
        elif cross_attn_type == 2:
            self.stft_posEmb = PositionalEncoding(dim_stft, max_len=3000, return_vec=True)
            self.rep_posEmb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)
            self.stft_pos_emb = cross_attn_layer(nhead, dim_stft//nhead, dim_stft//nhead, dim_stft, conformer_class, d_ffn)
            self.rep_pos_emb = cross_attn_layer(nhead, conformer_class//nhead, conformer_class//nhead, conformer_class, conformer_class, d_ffn)
            self.crossattn = cross_attn_layer(nhead, conformer_class//nhead, conformer_class//nhead, conformer_class, conformer_class, d_ffn)
            
        elif cross_attn_type == 3:
            self.rep_posEmb = PositionalEncoding(conformer_class, max_len=3000, return_vec=True)
            self.stft_posEmb = PositionalEncoding(dim_stft, max_len=3000, return_vec=True)
            self.stft_rep = cross_attn_layer(nhead, conformer_class//nhead, conformer_class//nhead, conformer_class, conformer_class, d_ffn)
            self.crossattn = cross_attn_layer(nhead, conformer_class//nhead, conformer_class//nhead, conformer_class, conformer_class, d_ffn)
            self.stft_pos_emb = cross_attn_layer(nhead, conformer_class//nhead, conformer_class//nhead, dim_stft, conformer_class, d_ffn)

        self.recover_type = recover_type
        if recover_type == 'upsample':
            self.rep_upconv = nn.Sequential(nn.Upsample(scale_factor=2),
                                       nn.Conv1d(conformer_class, conformer_class, kernel_size=3, padding='same'),
                                       nn.BatchNorm1d(conformer_class),
                                       nn.ReLU(),
                                       nn.Upsample(scale_factor=2),
                                       nn.Conv1d(conformer_class, conformer_class, kernel_size=3, padding='same'),
                                       nn.BatchNorm1d(conformer_class),
                                       nn.ReLU(),
                                       nn.Upsample(scale_factor=2),
                                       nn.Conv1d(conformer_class, conformer_class, kernel_size=3, padding='same'),
                                       nn.BatchNorm1d(conformer_class),
                                       nn.ReLU(),
                                       nn.Upsample(scale_factor=2),
                                       nn.Conv1d(conformer_class, conformer_class, kernel_size=3, padding='same'),
                                       nn.BatchNorm1d(conformer_class),
                                       nn.ReLU(),
                                       nn.Upsample(3000),
                                       nn.Conv1d(conformer_class, conformer_class, kernel_size=3, padding='same'),
                                       nn.BatchNorm1d(conformer_class),
                                       nn.ReLU(),)
            self.stft_upconv = nn.Sequential(nn.Upsample(scale_factor=2),
                                            nn.Conv1d(dim_stft, dim_stft, kernel_size=3, padding='same'),
                                            nn.BatchNorm1d(dim_stft),
                                            nn.ReLU(),
                                            nn.Upsample(scale_factor=2),
                                            nn.Conv1d(dim_stft, dim_stft, kernel_size=3, padding='same'),
                                            nn.BatchNorm1d(dim_stft),
                                            nn.ReLU(),
                                            nn.Upsample(scale_factor=2),
                                            nn.Conv1d(dim_stft, conformer_class, kernel_size=3, padding='same'),
                                            nn.BatchNorm1d(conformer_class),
                                            nn.ReLU(),
                                            nn.Upsample(3000),
                                            nn.Conv1d(conformer_class, conformer_class, kernel_size=3, padding='same'),
                                            nn.BatchNorm1d(conformer_class),
                                            nn.ReLU(),)

        # =========================================== #
        #                   Decoder                   #
        # =========================================== #    
        self.decoder_type = decoder_type
        
        if decoder_type == 'crossattn':
            self.decoder = nn.ModuleList([cross_attn_layer(nhead, conformer_class//nhead, conformer_class//nhead, conformer_class, conformer_class, d_ffn)
                                                for _ in range(dec_layers)]
                                                )
        elif decoder_type == 'MGAN':
            self.decoder = nn.ModuleList([MGAN(nhead, conformer_class//nhead, conformer_class//nhead, conformer_class, conformer_class, d_ffn, norm_type, l)
                                        for _ in range(dec_layers)])
        
        elif decoder_type == 'unet':
            self.down_decoder = nn.Sequential(nn.Conv1d(conformer_class, conformer_class*2, kernel_size=3, stride=2),
                                    #    nn.BatchNorm1d(conformer_class*2),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv1d(conformer_class*2, conformer_class*3, kernel_size=5, stride=3),
                                    #    nn.BatchNorm1d(conformer_class*3),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv1d(conformer_class*3, conformer_class*4, kernel_size=7, stride=3),
                                    #    nn.BatchNorm1d(conformer_class*4),
                                       nn.ReLU(),
                                        nn.Dropout(0.1),
                                       nn.Conv1d(conformer_class*4, conformer_class*5, kernel_size=9, stride=3),
                                    #    nn.BatchNorm1d(conformer_class*5),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       )
            self.decoder_layer = nn.TransformerEncoderLayer(d_model=conformer_class*5, nhead=nhead, dim_feedforward=d_ffn)
            self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=dec_layers)
            
            self.up_decoder = nn.Sequential(nn.Upsample(165),
                                      nn.Conv1d(conformer_class*5, conformer_class*4, kernel_size=3, padding='same'),
                                    #   nn.BatchNorm1d(conformer_class*4),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Upsample(499),
                                      nn.Conv1d(conformer_class*4, conformer_class*3, kernel_size=3, padding='same'),
                                    #   nn.BatchNorm1d(conformer_class*3),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Upsample(1499),
                                      nn.Conv1d(conformer_class*3, conformer_class*2, kernel_size=5, padding='same'),
                                    #   nn.BatchNorm1d(conformer_class*2),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Upsample(3000),
                                      nn.Conv1d(conformer_class*2, conformer_class*1, kernel_size=7, padding='same'),
                                    #   nn.BatchNorm1d(conformer_class),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),)
            
        elif decoder_type == 'residual_unet':
            self.decoder = Residual_Unet(conformer_class, nhead, d_ffn, dec_layers, res_dec)

        # =========================================== #
        #                    Output                   #
        # =========================================== #
        self.label_type = label_type
        
        if label_type == 'p':
            self.output = nn.Linear(conformer_class, 1)
            self.output_actfn = nn.Sigmoid()
        elif label_type == 'other':
            self.output = nn.Linear(conformer_class, 2)
            self.output_actfn = nn.Softmax(dim=-1)
        elif label_type == 'all':
            self.output = nn.ModuleList([nn.Linear(conformer_class, 1) for _ in range(3)])
            self.output_actfn = nn.Sigmoid()
        
    def forward(self, wave, stft):
        # wave: (batch, 3000, 12)
        wave = wave.permute(0,2,1)

        out, _ = self.conformer(wave, 3000)
  
        # temporal segmentation
        if self.seg_proj_type == 'crossattn':
            seg_pos_emb = self.seg_posEmb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
            seg_crossattn_out = self.seg_crossattn(seg_pos_emb, out, out)
            seg_out = self.seg_projector(seg_crossattn_out)
            seg_out = self.sigmoid(seg_out)
        elif self.seg_proj_type == 'upsample':
            seg_out = self.upconv(out.permute(0,2,1)).permute(0,2,1)
            seg_out = self.sigmoid(self.seg_projector(seg_out))
        else:
            seg_out = 0.0
        
        # cross_attention
        if self.cross_attn_type == 1:
            stft_posEmb = self.stft_posEmb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
            stft_rep_posEmb = self.stft_rep_posEmb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)

            stft_out = self.stft_pos_emb(stft_posEmb, stft, stft)
            stft_rep_out = self.stft_rep(stft_out, out, out) 
            crossattn_out = self.crossattn(stft_rep_posEmb, stft_rep_out, stft_rep_out)

        elif self.cross_attn_type == 2:
            if self.recover_type == 'upsample':
                stft_out = self.stft_upconv(stft.permute(0,2,1)).permute(0,2,1)
                rep_out = self.rep_upconv(out.permute(0,2,1)).permute(0,2,1)
            else:
                stft_posEmb = self.stft_posEmb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
                rep_posEmb = self.rep_posEmb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
                stft_out = self.stft_pos_emb(stft_posEmb, stft, stft)
                rep_out = self.rep_pos_emb(rep_posEmb, out, out)
            crossattn_out = self.crossattn(stft_out, rep_out, rep_out)
            
        elif self.cross_attn_type == 3:    
            rep_posEmb = self.rep_posEmb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
            stft_posEmb = self.stft_posEmb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)

            crossattn_out = self.crossattn(rep_posEmb, out, out)
            stft_out = self.stft_pos_emb(stft_posEmb, stft, stft)
        
        # decoder
        if self.decoder_type == 'unet':
            dec_tmp = self.down_decoder(crossattn_out.permute(0,2,1)).permute(0,2,1)
            dec_out = self.up_decoder(self.decoder(dec_tmp).permute(0,2,1)).permute(0,2,1)
        elif self.decoder_type == 'residual_unet':
            dec_out = self.decoder(crossattn_out.permute(0,2,1)).permute(0,2,1)
        else:
            for i, layer in enumerate(self.decoder):
                if i == 0:
                    if self.rep_KV:
                        dec_out = layer(crossattn_out, out, out)
                    else:
                        dec_out = layer(crossattn_out, crossattn_out, crossattn_out)
                else:
                    if self.rep_KV:
                        dec_out = layer(dec_out, out, out)
                    else:
                        dec_out = layer(dec_out, dec_out, dec_out)
        
        if self.cross_attn_type == 3:
            dec_out = self.stft_rep(stft_out, dec_out, dec_out)
            
        # output layer
        if self.label_type == 'p' or self.label_type == 'other':
            out = self.output_actfn(self.output(dec_out))
        elif self.label_type == 'all':
            out = []
            # 0: detection, 1: P-pahse, 2: S-phase
            for layer in self.output:
                out.append(self.output_actfn(layer(dec_out)))
        
        return seg_out, out

class REAL_GRADUATE(nn.Module):
    def __init__(self, d_model=12, conformer_class=16, d_ffn=64, n_head=2, enc_layers=2, wave_length=3000):
        super(REAL_GRADUATE, self).__init__()
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                    Encoder                    # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.shared_ENCconformer = Conformer(num_classes=d_model, input_dim=d_model*3, encoder_dim=d_ffn, 
                                          num_attention_heads=n_head, num_encoder_layers=enc_layers, subsample=False)
        
        self.downconv1 = nn.Sequential(nn.Conv1d(d_model, d_model*3, kernel_size=5, stride=3),
                                      nn.ReLU(),)
        self.downconv2 = nn.Sequential(nn.Conv1d(d_model, d_model*3, kernel_size=5, stride=3),
                                      nn.ReLU(),)
        
        self.downconv3 = nn.Sequential(nn.Conv1d(d_model, d_model*3, kernel_size=7, stride=5),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),)
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                   Bottleneck                  # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.bottleneck = nn.ModuleList([nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=3, padding='same'),
                            nn.BatchNorm1d(d_model),
                            nn.ReLU(),) for _ in range(5)])
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                      STFT                     # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        dim_stft = 32
        self.STFT_posemb = PositionalEncoding(dim_stft, max_len=wave_length, return_vec=True)
        
        self.STFT_crossattn = cross_attn_layer(n_head, dim_stft//n_head, dim_stft//n_head, 
                                             dim_stft, dim_stft, d_ffn)
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                 Phase Picking                 # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.STFT_to_picking = nn.Linear(dim_stft, d_model)
        
        self.shared_DECconformer = Conformer(num_classes=d_model, input_dim=d_model*3, encoder_dim=d_ffn, 
                                          num_attention_heads=n_head, num_encoder_layers=enc_layers, subsample=False)
        
        self.upconv1 = nn.Sequential(nn.Upsample(332),
                                    nn.Conv1d(d_model, d_model*3, kernel_size=3, padding='same'),)
        
        self.upconv2 = nn.Sequential(nn.Upsample(999),
                                    nn.Conv1d(d_model, d_model*3, kernel_size=5, padding='same'))
        
        self.upconv3 = nn.Sequential(nn.Upsample(3000),
                                    nn.Conv1d(d_model, d_model*3, kernel_size=7, padding='same'),
                                    nn.Dropout(0.1),)
        
        self.picking_out = nn.Sequential(nn.Linear(d_model, 3),
                                        nn.Sigmoid(),)
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #             Temporal Segmentation             # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.TS_posemb = PositionalEncoding(d_model, max_len=wave_length, return_vec=True)
        
        self.TS_crossattn = cross_attn_layer(n_head, d_model//n_head, d_model//n_head, 
                                             d_model, d_model, d_ffn)
    
        self.TS_out = nn.Conv1d(d_model, 1, kernel_size=5, padding='same')
        self.STFT_to_TS = nn.Linear(dim_stft, d_model)
        self.sigmoid = nn.Sigmoid()
                                        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #              Magnitude estimation             # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.mag_lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True, dropout=0.1, num_layers=2)

        self.fft_converter = nn.Sequential(nn.Conv1d(1, d_model//2, kernel_size=3, stride=2),
                                          nn.BatchNorm1d(d_model//2),
                                          nn.ReLU(),
                                          nn.Conv1d(d_model//2, d_model, kernel_size=5, stride=3),
                                          nn.BatchNorm1d(d_model),
                                          nn.ReLU(),
                                          nn.Conv1d(d_model, d_model*2, kernel_size=5, stride=3),
                                          nn.BatchNorm1d(d_model*2),
                                          nn.ReLU(),)
        
        self.mag_fft_crossattn = cross_attn_layer(n_head, d_model*2//n_head, d_model*2//n_head, 
                                             d_model*2, d_model*2, d_ffn)
        
        self.mag_std_scalar = ScalarProjector(output_dim=66)
        self.mag_mean_scalar = ScalarProjector(output_dim=1)

        self.mag_selfattn = cross_attn_layer(n_head, d_model*2//n_head, d_model*2//n_head, 
                                             d_model*2, d_model*2, d_ffn)

        self.mag_out = nn.Sequential(nn.Conv1d(24, 16, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(16),
                                      nn.MaxPool1d(3),
                                      nn.ReLU(),
                                      nn.Conv1d(16, 8, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(8),
                                      nn.MaxPool1d(3),
                                      nn.ReLU(),
                                      nn.Conv1d(8, 4, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(4),
                                      nn.MaxPool1d(3),
                                      nn.ReLU(),
                                      nn.Conv1d(4, 1, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(1),
                                      nn.MaxPool1d(2),
                                      nn.ReLU(),)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, wave, stft, fft, mean_std):
        # print('original: ', wave.shape)
        
        # downsampling encoder
        downconv1 = self.downconv1(wave)
        # print('downconv1: ', downconv1.shape)
        
        enc_conformer1, _ = self.shared_ENCconformer(downconv1.permute(0,2,1), downconv1.shape[-1])
        # print('conformer1: ', enc_conformer1.shape)
        
        downconv2 = self.downconv2(enc_conformer1.permute(0,2,1))
        # print('downconv2: ', downconv2.shape)
        
        enc_conformer2, _ = self.shared_ENCconformer(downconv2.permute(0,2,1), downconv2.shape[-1])
        # print('conformer2: ', enc_conformer2.shape)
        
        downconv3 = self.downconv3(enc_conformer2.permute(0,2,1))
        # print('downconv3: ', downconv3.shape)
        
        enc_conformer3, _ = self.shared_ENCconformer(downconv3.permute(0,2,1), downconv3.shape[-1])
        # print('conformer3: ', enc_conformer3.shape)
        
        for idx, layer in enumerate(self.bottleneck):
            if idx == 0:
                bottleneck_out = layer(enc_conformer3.permute(0,2,1))
            else:
                bottleneck_out = layer(enc_conformer3.permute(0,2,1)) + bottleneck_out
                
        # print('bottleneck_out: ', bottleneck_out.shape)
        
        # recover the length of STFT to orignal length of trace
        stft_posemb = self.STFT_posemb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
        stft_vector = self.STFT_crossattn(stft_posemb, stft, stft)
        # print('stft_vector: ', stft_vector.shape)
        
        # Temporal segmentation
        ts_posemb = self.TS_posemb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
        ts_out = self.TS_crossattn(ts_posemb, bottleneck_out.permute(0,2,1), bottleneck_out.permute(0,2,1))
        ts_out = self.STFT_to_TS(stft_vector) + ts_out
        ts_out = self.TS_out(ts_out.permute(0,2,1)).permute(0,2,1)
        ts_out = self.sigmoid(ts_out)
        # print('ts_out: ', ts_out.shape)
        
        # Magnitude estimation

        # mean_std: (batch, 6)
        mean, std = mean_std[:, :3], mean_std[:, 3:]
       
        mean_scalar = self.mag_mean_scalar(wave[:, :3], mean)
        std_scalar = self.mag_std_scalar(wave[:, :3], std)

        fft_vector = self.fft_converter(fft)[:, :, 8:-8].permute(0,2,1)
        mag_hidden_state, _ = self.mag_lstm(enc_conformer3)
        mag_out = self.mag_fft_crossattn(fft_vector, mag_hidden_state, mag_hidden_state)
        mag_out = self.mag_selfattn(mag_out, mag_out, mag_out, mean=mean_scalar, std=std_scalar)
       
        mag_out = self.mag_out(mag_out.permute(0,2,1))
        # print('mag_out: ', mag_out.shape)        
        
        # P/S phase picking
        upconv1 = self.upconv1(bottleneck_out)
        # print('upconv1: ', upconv1.shape)
        
        dec_conformer1, _ = self.shared_DECconformer(upconv1.permute(0,2,1), upconv1.shape[-1])
        # print('dec_conformer1: ', dec_conformer1.shape)
        
        upconv2 = self.upconv2((dec_conformer1+enc_conformer2).permute(0,2,1))
        # print('upconv2: ', upconv2.shape)
        
        dec_conformer2, _ = self.shared_DECconformer(upconv2.permute(0,2,1), upconv2.shape[-1])
        # print('dec_conformer2: ', dec_conformer2.shape)
        
        upconv3 = self.upconv3((dec_conformer2+enc_conformer1).permute(0,2,1))
        # print('upconv1: ', upconv3.shape)
        
        dec_conformer3, _ = self.shared_DECconformer(upconv3.permute(0,2,1), upconv3.shape[-1])
        # print('dec_conformer3: ', dec_conformer3.shape)
        
        picking_out = self.picking_out(dec_conformer3+self.STFT_to_picking(stft_vector))
        # print('picking_out: ', picking_out.shape)
        
        return ts_out, mag_out, picking_out     

class REAL_GRADUATE_noNorm(nn.Module):
    def __init__(self, d_model=12, conformer_class=16, d_ffn=64, n_head=2, enc_layers=2, wave_length=3000):
        super(REAL_GRADUATE_noNorm, self).__init__()
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                    Encoder                    # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.shared_ENCconformer = Conformer(num_classes=d_model, input_dim=d_model*3, encoder_dim=d_ffn, 
                                          num_attention_heads=n_head, num_encoder_layers=enc_layers, subsample=False)
        
        self.downconv1 = nn.Sequential(nn.Conv1d(d_model, d_model*3, kernel_size=5, stride=3),
                                      nn.ReLU(),)
        self.downconv2 = nn.Sequential(nn.Conv1d(d_model, d_model*3, kernel_size=5, stride=3),
                                      nn.ReLU(),)
        
        self.downconv3 = nn.Sequential(nn.Conv1d(d_model, d_model*3, kernel_size=7, stride=5),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),)
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                   Bottleneck                  # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.bottleneck = nn.ModuleList([nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=3, padding='same'),
                            nn.BatchNorm1d(d_model),
                            nn.ReLU(),) for _ in range(5)])
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                      STFT                     # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        dim_stft = 32
        self.STFT_posemb = PositionalEncoding(dim_stft, max_len=wave_length, return_vec=True)
        
        self.STFT_crossattn = cross_attn_layer(n_head, dim_stft//n_head, dim_stft//n_head, 
                                             dim_stft, dim_stft, d_ffn)
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                 Phase Picking                 # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.STFT_to_picking = nn.Linear(dim_stft, d_model)
        
        self.shared_DECconformer = Conformer(num_classes=d_model, input_dim=d_model*3, encoder_dim=d_ffn, 
                                          num_attention_heads=n_head, num_encoder_layers=enc_layers, subsample=False)
        
        self.upconv1 = nn.Sequential(nn.Upsample(332),
                                    nn.Conv1d(d_model, d_model*3, kernel_size=3, padding='same'),)
        
        self.upconv2 = nn.Sequential(nn.Upsample(999),
                                    nn.Conv1d(d_model, d_model*3, kernel_size=5, padding='same'))
        
        self.upconv3 = nn.Sequential(nn.Upsample(3000),
                                    nn.Conv1d(d_model, d_model*3, kernel_size=7, padding='same'),
                                    nn.Dropout(0.1),)
        
        self.picking_out = nn.Sequential(nn.Linear(d_model, 3),
                                        nn.Sigmoid(),)
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #             Temporal Segmentation             # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.TS_posemb = PositionalEncoding(d_model, max_len=wave_length, return_vec=True)
        
        self.TS_crossattn = cross_attn_layer(n_head, d_model//n_head, d_model//n_head, 
                                             d_model, d_model, d_ffn)
    
        self.TS_out = nn.Conv1d(d_model, 1, kernel_size=5, padding='same')
        self.STFT_to_TS = nn.Linear(dim_stft, d_model)
        self.sigmoid = nn.Sigmoid()
                                        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #              Magnitude estimation             # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.mag_lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True, dropout=0.1, num_layers=2)

        self.fft_converter = nn.Sequential(nn.Conv1d(1, d_model//2, kernel_size=3, stride=2),
                                          nn.BatchNorm1d(d_model//2),
                                          nn.ReLU(),
                                          nn.Conv1d(d_model//2, d_model, kernel_size=5, stride=3),
                                          nn.BatchNorm1d(d_model),
                                          nn.ReLU(),
                                          nn.Conv1d(d_model, d_model*2, kernel_size=5, stride=3),
                                          nn.BatchNorm1d(d_model*2),
                                          nn.ReLU(),)
        
        self.mag_fft_crossattn = cross_attn_layer(n_head, d_model*2//n_head, d_model*2//n_head, 
                                             d_model*2, d_model*2, d_ffn)

        self.mag_selfattn = cross_attn_layer(n_head, d_model*2//n_head, d_model*2//n_head, 
                                             d_model*2, d_model*2, d_ffn)

        self.mag_out = nn.Sequential(nn.Conv1d(24, 16, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(16),
                                      nn.MaxPool1d(3),
                                      nn.ReLU(),
                                      nn.Conv1d(16, 8, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(8),
                                      nn.MaxPool1d(3),
                                      nn.ReLU(),
                                      nn.Conv1d(8, 4, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(4),
                                      nn.MaxPool1d(3),
                                      nn.ReLU(),
                                      nn.Conv1d(4, 1, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(1),
                                      nn.MaxPool1d(2),
                                      nn.ReLU(),)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, wave, stft, fft):
        # print('original: ', wave.shape)
        
        # downsampling encoder
        downconv1 = self.downconv1(wave)
        # print('downconv1: ', downconv1.shape)
        
        enc_conformer1, _ = self.shared_ENCconformer(downconv1.permute(0,2,1), downconv1.shape[-1])
        # print('conformer1: ', enc_conformer1.shape)
        
        downconv2 = self.downconv2(enc_conformer1.permute(0,2,1))
        # print('downconv2: ', downconv2.shape)
        
        enc_conformer2, _ = self.shared_ENCconformer(downconv2.permute(0,2,1), downconv2.shape[-1])
        # print('conformer2: ', enc_conformer2.shape)
        
        downconv3 = self.downconv3(enc_conformer2.permute(0,2,1))
        # print('downconv3: ', downconv3.shape)
        
        enc_conformer3, _ = self.shared_ENCconformer(downconv3.permute(0,2,1), downconv3.shape[-1])
        # print('conformer3: ', enc_conformer3.shape)
        
        for idx, layer in enumerate(self.bottleneck):
            if idx == 0:
                bottleneck_out = layer(enc_conformer3.permute(0,2,1))
            else:
                bottleneck_out = layer(enc_conformer3.permute(0,2,1)) + bottleneck_out
                
        # print('bottleneck_out: ', bottleneck_out.shape)
        
        # recover the length of STFT to orignal length of trace
        stft_posemb = self.STFT_posemb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
        stft_vector = self.STFT_crossattn(stft_posemb, stft, stft)
        # print('stft_vector: ', stft_vector.shape)
        
        # Temporal segmentation
        ts_posemb = self.TS_posemb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
        ts_out = self.TS_crossattn(ts_posemb, bottleneck_out.permute(0,2,1), bottleneck_out.permute(0,2,1))
        ts_out = self.STFT_to_TS(stft_vector) + ts_out
        ts_out = self.TS_out(ts_out.permute(0,2,1)).permute(0,2,1)
        ts_out = self.sigmoid(ts_out)
        # print('ts_out: ', ts_out.shape)
        
        # Magnitude estimation
        fft_vector = self.fft_converter(fft)[:, :, 8:-8].permute(0,2,1)
        mag_hidden_state, _ = self.mag_lstm(enc_conformer3)
        mag_out = self.mag_fft_crossattn(fft_vector, mag_hidden_state, mag_hidden_state)
        mag_out = self.mag_selfattn(mag_out, mag_out, mag_out)
       
        mag_out = self.mag_out(mag_out.permute(0,2,1))
        # print('mag_out: ', mag_out.shape)        
        
        # P/S phase picking
        upconv1 = self.upconv1(bottleneck_out)
        # print('upconv1: ', upconv1.shape)
        
        dec_conformer1, _ = self.shared_DECconformer(upconv1.permute(0,2,1), upconv1.shape[-1])
        # print('dec_conformer1: ', dec_conformer1.shape)
        
        upconv2 = self.upconv2((dec_conformer1+enc_conformer2).permute(0,2,1))
        # print('upconv2: ', upconv2.shape)
        
        dec_conformer2, _ = self.shared_DECconformer(upconv2.permute(0,2,1), upconv2.shape[-1])
        # print('dec_conformer2: ', dec_conformer2.shape)
        
        upconv3 = self.upconv3((dec_conformer2+enc_conformer1).permute(0,2,1))
        # print('upconv1: ', upconv3.shape)
        
        dec_conformer3, _ = self.shared_DECconformer(upconv3.permute(0,2,1), upconv3.shape[-1])
        # print('dec_conformer3: ', dec_conformer3.shape)
        
        picking_out = self.picking_out(dec_conformer3+self.STFT_to_picking(stft_vector))
        # print('picking_out: ', picking_out.shape)
        
        return ts_out, mag_out, picking_out     

class REAL_GRADUATE_doubleWave(nn.Module):
    def __init__(self, d_model=12, conformer_class=16, d_ffn=64, n_head=2, enc_layers=2, wave_length=3000):
        super(REAL_GRADUATE_doubleWave, self).__init__()
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                    Encoder                    # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.shared_ENCconformer = Conformer(num_classes=d_model, input_dim=d_model*3, encoder_dim=d_ffn, 
                                          num_attention_heads=n_head, num_encoder_layers=enc_layers, subsample=False)
        
        self.downconv1 = nn.Sequential(nn.Conv1d(d_model, d_model*3, kernel_size=5, stride=3),
                                      nn.ReLU(),)
        self.downconv2 = nn.Sequential(nn.Conv1d(d_model, d_model*3, kernel_size=5, stride=3),
                                      nn.ReLU(),)
        
        self.downconv3 = nn.Sequential(nn.Conv1d(d_model, d_model*3, kernel_size=7, stride=5),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),)
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                   Bottleneck                  # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.bottleneck = nn.ModuleList([nn.Sequential(nn.Conv1d(d_model, d_model, kernel_size=3, padding='same'),
                            nn.BatchNorm1d(d_model),
                            nn.ReLU(),) for _ in range(5)])
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                      STFT                     # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        dim_stft = 32
        self.STFT_posemb = PositionalEncoding(dim_stft, max_len=wave_length, return_vec=True)
        
        self.STFT_crossattn = cross_attn_layer(n_head, dim_stft//n_head, dim_stft//n_head, 
                                             dim_stft, dim_stft, d_ffn)
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #                 Phase Picking                 # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.STFT_to_picking = nn.Linear(dim_stft, d_model)
        
        self.shared_DECconformer = Conformer(num_classes=d_model, input_dim=d_model*3, encoder_dim=d_ffn, 
                                          num_attention_heads=n_head, num_encoder_layers=enc_layers, subsample=False)
        
        self.upconv1 = nn.Sequential(nn.Upsample(332),
                                    nn.Conv1d(d_model, d_model*3, kernel_size=3, padding='same'),)
        
        self.upconv2 = nn.Sequential(nn.Upsample(999),
                                    nn.Conv1d(d_model, d_model*3, kernel_size=5, padding='same'))
        
        self.upconv3 = nn.Sequential(nn.Upsample(3000),
                                    nn.Conv1d(d_model, d_model*3, kernel_size=7, padding='same'),
                                    nn.Dropout(0.1),)
        
        self.picking_out = nn.Sequential(nn.Linear(d_model, 3),
                                        nn.Sigmoid(),)
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #             Temporal Segmentation             # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.TS_posemb = PositionalEncoding(d_model, max_len=wave_length, return_vec=True)
        
        self.TS_crossattn = cross_attn_layer(n_head, d_model//n_head, d_model//n_head, 
                                             d_model, d_model, d_ffn)
    
        self.TS_out = nn.Conv1d(d_model, 1, kernel_size=5, padding='same')
        self.STFT_to_TS = nn.Linear(dim_stft, d_model)
        self.sigmoid = nn.Sigmoid()
                                        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #              Magnitude estimation             # 
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.mag_lstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True, dropout=0.1, num_layers=2)

        self.fft_converter = nn.Sequential(nn.Conv1d(1, d_model//2, kernel_size=3, stride=2),
                                          nn.BatchNorm1d(d_model//2),
                                          nn.ReLU(),
                                          nn.Conv1d(d_model//2, d_model, kernel_size=5, stride=3),
                                          nn.BatchNorm1d(d_model),
                                          nn.ReLU(),
                                          nn.Conv1d(d_model, d_model*2, kernel_size=5, stride=3),
                                          nn.BatchNorm1d(d_model*2),
                                          nn.ReLU(),)
        
        self.mag_fft_crossattn = cross_attn_layer(n_head, d_model*2//n_head, d_model*2//n_head, 
                                             d_model*2, d_model*2, d_ffn)

        self.mag_selfattn = cross_attn_layer(n_head, d_model*2//n_head, d_model*2//n_head, 
                                             d_model*2, d_model*2, d_ffn)

        self.mag_out = nn.Sequential(nn.Conv1d(48, 32, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(32),
                                      nn.MaxPool1d(3),
                                      nn.ReLU(),
                                      nn.Conv1d(32, 16, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(16),
                                      nn.MaxPool1d(3),
                                      nn.ReLU(),
                                      nn.Conv1d(16, 8, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(8),
                                      nn.MaxPool1d(3),
                                      nn.ReLU(),
                                      nn.Conv1d(8, 1, kernel_size=3, padding='same'),
                                      nn.BatchNorm1d(1),
                                      nn.MaxPool1d(2),
                                      nn.ReLU(),)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, wave, stft, fft):
        # print('original: ', wave.shape)
        
        # downsampling encoder
        downconv1 = self.downconv1(wave)
        # print('downconv1: ', downconv1.shape)
        
        enc_conformer1, _ = self.shared_ENCconformer(downconv1.permute(0,2,1), downconv1.shape[-1])
        # print('conformer1: ', enc_conformer1.shape)
        
        downconv2 = self.downconv2(enc_conformer1.permute(0,2,1))
        # print('downconv2: ', downconv2.shape)
        
        enc_conformer2, _ = self.shared_ENCconformer(downconv2.permute(0,2,1), downconv2.shape[-1])
        # print('conformer2: ', enc_conformer2.shape)
        
        downconv3 = self.downconv3(enc_conformer2.permute(0,2,1))
        # print('downconv3: ', downconv3.shape)
        
        enc_conformer3, _ = self.shared_ENCconformer(downconv3.permute(0,2,1), downconv3.shape[-1])
        # print('conformer3: ', enc_conformer3.shape)
        
        for idx, layer in enumerate(self.bottleneck):
            if idx == 0:
                bottleneck_out = layer(enc_conformer3.permute(0,2,1))
            else:
                bottleneck_out = layer(enc_conformer3.permute(0,2,1)) + bottleneck_out
                
        # print('bottleneck_out: ', bottleneck_out.shape)
        
        # recover the length of STFT to orignal length of trace
        stft_posemb = self.STFT_posemb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
        stft_vector = self.STFT_crossattn(stft_posemb, stft, stft)
        # print('stft_vector: ', stft_vector.shape)
        
        # Temporal segmentation
        ts_posemb = self.TS_posemb(wave).unsqueeze(0).repeat(wave.size(0), 1, 1)
        ts_out = self.TS_crossattn(ts_posemb, bottleneck_out.permute(0,2,1), bottleneck_out.permute(0,2,1))
        ts_out = self.STFT_to_TS(stft_vector) + ts_out
        ts_out = self.TS_out(ts_out.permute(0,2,1)).permute(0,2,1)
        ts_out = self.sigmoid(ts_out)
        # print('ts_out: ', ts_out.shape)
        
        # Magnitude estimation
        fft_vector = self.fft_converter(fft)[:, :, 8:-8].permute(0,2,1)
        mag_hidden_state, _ = self.mag_lstm(enc_conformer3)
        mag_out = self.mag_fft_crossattn(fft_vector, mag_hidden_state, mag_hidden_state)
        mag_out = self.mag_selfattn(mag_out, mag_out, mag_out)
       
        mag_out = self.mag_out(mag_out.permute(0,2,1))
        # print('mag_out: ', mag_out.shape)        
        
        # P/S phase picking
        upconv1 = self.upconv1(bottleneck_out)
        # print('upconv1: ', upconv1.shape)
        
        dec_conformer1, _ = self.shared_DECconformer(upconv1.permute(0,2,1), upconv1.shape[-1])
        # print('dec_conformer1: ', dec_conformer1.shape)
        
        upconv2 = self.upconv2((dec_conformer1+enc_conformer2).permute(0,2,1))
        # print('upconv2: ', upconv2.shape)
        
        dec_conformer2, _ = self.shared_DECconformer(upconv2.permute(0,2,1), upconv2.shape[-1])
        # print('dec_conformer2: ', dec_conformer2.shape)
        
        upconv3 = self.upconv3((dec_conformer2+enc_conformer1).permute(0,2,1))
        # print('upconv1: ', upconv3.shape)
        
        dec_conformer3, _ = self.shared_DECconformer(upconv3.permute(0,2,1), upconv3.shape[-1])
        # print('dec_conformer3: ', dec_conformer3.shape)
        
        picking_out = self.picking_out(dec_conformer3+self.STFT_to_picking(stft_vector))
        # print('picking_out: ', picking_out.shape)
        
        return ts_out, mag_out, picking_out