import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
sys.path.append('/home/weiwei/disk4/picking_baseline')
from conformer import *

class Encoder(nn.Module):
    """
    Encoder stack
    """

    def __init__(self, input_channels, filters, kernel_sizes, in_samples):
        super().__init__()

        convs = []
        pools = []
        self.paddings = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )

            # To be consistent with the behaviour in tensorflow,
            # padding needs to be added for odd numbers of input_samples
            padding = in_samples % 2

            # Padding for MaxPool1d needs to be handled manually to conform with tf padding
            self.paddings.append(padding)
            pools.append(nn.MaxPool1d(2, padding=0))
            in_samples = (in_samples + padding) // 2

        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)

    def forward(self, x):
        for conv, pool, padding in zip(self.convs, self.pools, self.paddings):
            x = torch.relu(conv(x))
            if padding != 0:
                # Only pad right, use -1e10 as negative infinity
                x = F.pad(x, (0, padding), "constant", -1e10)
            x = pool(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        kernel_sizes,
        out_samples,
        original_compatible=False,
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.original_compatible = original_compatible

        # We need to trim off the final sample sometimes to get to the right number of output samples
        self.crops = []
        current_samples = out_samples
        for i, _ in enumerate(filters):
            padding = current_samples % 2
            current_samples = (current_samples + padding) // 2
            if padding == 1:
                self.crops.append(len(filters) - 1 - i)

        convs = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )

        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = self.upsample(x)

            if self.original_compatible:
                if i == 3:
                    x = x[:, :, 1:-1]
            else:
                if i in self.crops:
                    x = x[:, :, :-1]

            x = F.relu(conv(x))

        return x


class ResCNNStack(nn.Module):
    def __init__(self, kernel_sizes, filters, drop_rate):
        super().__init__()

        members = []
        for ker in kernel_sizes:
            members.append(ResCNNBlock(filters, ker, drop_rate))

        self.members = nn.ModuleList(members)

    def forward(self, x):
        for member in self.members:
            x = member(x)

        return x


class ResCNNBlock(nn.Module):
    def __init__(self, filters, ker, drop_rate):
        super().__init__()

        self.manual_padding = False
        if ker == 3:
            padding = 1
        else:
            # ker == 2
            # Manual padding emulate the padding in tensorflow
            self.manual_padding = True
            padding = 0

        self.dropout = SpatialDropout1d(drop_rate)

        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters, ker, padding=padding)

        self.norm2 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv2 = nn.Conv1d(filters, filters, ker, padding=padding)

    def forward(self, x):
        y = self.norm1(x)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv1(y)

        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv2(y)

        return x + y


class BiLSTMStack(nn.Module):
    def __init__(
        self, blocks, input_size, drop_rate, hidden_size=16, original_compatible=False
    ):
        super().__init__()

        # First LSTM has a different input size as the subsequent ones
        self.members = nn.ModuleList(
            [
                BiLSTMBlock(
                    input_size,
                    hidden_size,
                    drop_rate,
                    original_compatible=original_compatible,
                )
            ]
            + [
                BiLSTMBlock(
                    hidden_size,
                    hidden_size,
                    drop_rate,
                    original_compatible=original_compatible,
                )
                for _ in range(blocks - 1)
            ]
        )

    def forward(self, x):
        for member in self.members:
            x = member(x)
        return x


class BiLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, drop_rate, original_compatible=False):
        super().__init__()

        if original_compatible == "conservative":
            # The non-conservative model uses a sigmoid activiation as handled by the base nn.LSTM
            self.lstm = CustomLSTM(ActivationLSTMCell, input_size, hidden_size)
        elif original_compatible == "non-conservative":
            self.lstm = CustomLSTM(
                ActivationLSTMCell,
                input_size,
                hidden_size,
                gate_activation=torch.sigmoid,
            )
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(drop_rate)
        self.conv = nn.Conv1d(2 * hidden_size, hidden_size, 1)
        self.norm = nn.BatchNorm1d(hidden_size, eps=1e-3)

    def forward(self, x):
        x = x.permute(
            2, 0, 1
        )  # From batch, channels, sequence to sequence, batch, channels
        x = self.lstm(x)[0]
        x = self.dropout(x)
        x = x.permute(
            1, 2, 0
        )  # From sequence, batch, channels to batch, channels, sequence
        x = self.conv(x)
        x = self.norm(x)
        return x
    
class SpatialDropout1d(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = x.unsqueeze(dim=-1)  # Add fake dimension
        x = self.dropout(x)
        x = x.squeeze(dim=-1)  # Remove fake dimension
        return x
    
class Decoder(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        kernel_sizes,
        out_samples,
        original_compatible=False,
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.original_compatible = original_compatible

        # We need to trim off the final sample sometimes to get to the right number of output samples
        self.crops = []
        current_samples = out_samples
        for i, _ in enumerate(filters):
            padding = current_samples % 2
            current_samples = (current_samples + padding) // 2
            if padding == 1:
                self.crops.append(len(filters) - 1 - i)

        convs = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )

        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = self.upsample(x)

            if self.original_compatible:
                if i == 3:
                    x = x[:, :, 1:-1]
            else:
                if i in self.crops:
                    x = x[:, :, :-1]

            x = F.relu(conv(x))

        return x
    
class SeqSelfAttention(nn.Module):
    """
    Additive self attention
    """

    def __init__(self, input_size, units=32, attention_width=None, eps=1e-5):
        super().__init__()
        self.attention_width = attention_width

        self.Wx = nn.Parameter(uniform(-0.02, 0.02, input_size, units))
        self.Wt = nn.Parameter(uniform(-0.02, 0.02, input_size, units))
        self.bh = nn.Parameter(torch.zeros(units))

        self.Wa = nn.Parameter(uniform(-0.02, 0.02, units, 1))
        self.ba = nn.Parameter(torch.zeros(1))

        self.eps = eps

    def forward(self, x):
        # x.shape == (batch, channels, time)

        x = x.permute(0, 2, 1)  # to (batch, time, channels)

        q = torch.unsqueeze(
            torch.matmul(x, self.Wt), 2
        )  # Shape (batch, time, 1, channels)
        k = torch.unsqueeze(
            torch.matmul(x, self.Wx), 1
        )  # Shape (batch, 1, time, channels)

        h = torch.tanh(q + k + self.bh)

        # Emissions
        e = torch.squeeze(
            torch.matmul(h, self.Wa) + self.ba, -1
        )  # Shape (batch, time, time)

        # This is essentially softmax with an additional attention component.
        e = (
            e - torch.max(e, dim=-1, keepdim=True).values
        )  # In versions <= 0.2.1 e was incorrectly normalized by max(x)
        e = torch.exp(e)
        if self.attention_width is not None:
            lower = (
                torch.arange(0, e.shape[1], device=e.device) - self.attention_width // 2
            )
            upper = lower + self.attention_width
            indices = torch.unsqueeze(torch.arange(0, e.shape[1], device=e.device), 1)
            mask = torch.logical_and(lower <= indices, indices < upper)
            e = torch.where(mask, e, torch.zeros_like(e))

        a = e / (torch.sum(e, dim=-1, keepdim=True) + self.eps)

        v = torch.matmul(a, x)

        v = v.permute(0, 2, 1)  # to (batch, channels, time)

        return v, a
    
def uniform(a, b, *args):
    return a + (b - a) * torch.rand(*args)


class LayerNormalization(nn.Module):
    def __init__(self, filters, eps=1e-14):
        super().__init__()

        gamma = torch.ones(filters, 1)
        self.gamma = nn.Parameter(gamma)
        beta = torch.zeros(filters, 1)
        self.beta = nn.Parameter(beta)
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        var = torch.mean((x - mean) ** 2, 1, keepdim=True) + self.eps
        std = torch.sqrt(var)
        outputs = (x - mean) / std

        outputs = outputs * self.gamma
        outputs = outputs + self.beta

        return outputs


class FeedForward(nn.Module):
    def __init__(self, io_size, drop_rate, hidden_size=128):
        super().__init__()

        self.lin1 = nn.Linear(io_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, io_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # To (batch, time, channel)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        x = x.permute(0, 2, 1)  # To (batch, channel, time)

        return x
    
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
        output = torch.matmul(attn, v)
        
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        
        output = self.dropout(self.fc(output))
        output += residual
        
        output = self.layer_norm(output)
        
        return output

class cross_attn_layer(nn.Module):
    def __init__(self, nhead, d_k, d_v, d_model, d_ffn):
        super(cross_attn_layer, self).__init__()
        
        self.cross_attn = cross_attn(nhead=nhead, d_k=d_k, d_v=d_v, d_model=d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn),
                                    nn.ReLU(),
                                    nn.Linear(d_ffn, d_model),
                                )   
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, q, k, v):
        out_attn = self.cross_attn(q, k, v)
            
        out = self.layer_norm(self.ffn(out_attn) + out_attn)
        out = self.dropout(out)
        
        return out

class Transformer(nn.Module):
    def __init__(self, input_size, drop_rate, attention_width=None, eps=1e-5):
        super().__init__()

        self.attention = SeqSelfAttention(
            input_size, attention_width=attention_width, eps=eps
        )
        self.norm1 = LayerNormalization(input_size)
        self.ff = FeedForward(input_size, drop_rate)
        self.norm2 = LayerNormalization(input_size)

    def forward(self, x):
        y, weight = self.attention(x)
        y = x + y
        y = self.norm1(y)
        y2 = self.ff(y)
        y2 = y + y2
        y2 = self.norm2(y2)

        return y2, weight

class TSFC_Unet(nn.Module):
    def __init__(self, d_model=3, lstm_blocks=3, hidden_size=16, isConformer=False):
        super(TSFC_Unet, self).__init__()
        
        # Parameters from EQTransformer repository
        self.filters = [
            8,
            16,
            16,
            32,
            32,
            64,
            64,
        ]  # Number of filters for the convolutions
        self.kernel_sizes = [11, 9, 7, 7, 5, 5, 3]  # Kernel sizes for the convolutions
        self.res_cnn_kernels = [3, 3, 3, 3, 2, 3, 2]

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #               Feature Extractor               #
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        # Encoder stack
        self.encoder = Encoder(
            input_channels=d_model,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=3000,
        )

        # Res CNN Stack
        self.res_cnn_stack = ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=0.1,
        )

        # BiLSTM stack
        self.bi_lstm_stack = BiLSTMStack(
            blocks=lstm_blocks,
            input_size=self.filters[-1],
            drop_rate=0.1,
            hidden_size=hidden_size,
        )
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #             Temporal Segmentation             #
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.upconv = nn.Sequential(nn.Upsample(scale_factor=3, mode="nearest"),
                                    nn.Conv1d(hidden_size, hidden_size, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.Upsample(scale_factor=3, mode="nearest"),
                                    nn.Conv1d(hidden_size, hidden_size, 3, padding='same'),
                                    nn.ReLU(),
                                    nn.Upsample(scale_factor=3, mode="nearest"),
                                    nn.Conv1d(hidden_size, hidden_size, 5, padding='same'),
                                    nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode="nearest"),
                                    nn.Conv1d(hidden_size, hidden_size, 5, padding='same'),
                                    nn.ReLU(),
                                    nn.Upsample(scale_factor=2, mode="nearest"),
                                    nn.Conv1d(hidden_size, hidden_size, 5, padding='same'),
                                    nn.ReLU(),
                                    nn.Upsample(3000),
                                    nn.Conv1d(hidden_size, hidden_size, 5, padding='same'),
                                    nn.ReLU(),
                                    )
        self.seg_projector = nn.Sequential(nn.Linear(hidden_size, hidden_size*3),
                                               nn.ReLU(),
                                               nn.Linear(hidden_size*3, 1)) 
        self.sigmoid = nn.Sigmoid()
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #               Bottleneck layer                #
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.isConformer = isConformer
        if not isConformer:
            # Global attention - two transformers
            self.transformer_d0 = Transformer(
                input_size=16, drop_rate=0.1, eps=1e-7
            )
            self.transformer_d = Transformer(
                input_size=16, drop_rate=0.1, eps=1e-7
            )
        else:
            self.conformer = Conformer(num_classes=16, input_dim=16, encoder_dim=128, num_attention_heads=1, num_encoder_layers=2, subsample=False)

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #             STFT cross-attention              #
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        self.stft_cross = nn.ModuleList()
        for _ in range(3):
            self.stft_cross.append(cross_attn_layer(4, 16//4, 16//4, 16, 128))
        
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        #              Multi-task decoder               #
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        # Detection decoder and final Conv
        self.decoder_d = Decoder(
            input_channels=16,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=3000,
        )
        self.conv_d = nn.Conv1d(
            in_channels=self.filters[0], out_channels=1, kernel_size=11, padding=5
        )
        
        # Picking branches
        self.pick_lstms = []
        self.pick_attentions = []
        self.pick_decoders = []
        self.pick_convs = []
        self.dropout = nn.Dropout(0.1)

        for _ in range(2):
            lstm = nn.LSTM(16, 16, bidirectional=False)
            self.pick_lstms.append(lstm)

            attention = SeqSelfAttention(input_size=16, attention_width=3, eps=1e-7)
            self.pick_attentions.append(attention)

            decoder = Decoder(
                input_channels=16,
                filters=self.filters[::-1],
                kernel_sizes=self.kernel_sizes[::-1],
                out_samples=3000,
            )
            self.pick_decoders.append(decoder)

            conv = nn.Conv1d(
                in_channels=self.filters[0], out_channels=1, kernel_size=11, padding=5
            )
            self.pick_convs.append(conv)

        self.pick_lstms = nn.ModuleList(self.pick_lstms)
        self.pick_attentions = nn.ModuleList(self.pick_attentions)
        self.pick_decoders = nn.ModuleList(self.pick_decoders)
        self.pick_convs = nn.ModuleList(self.pick_convs)
        
        # Magnitude estimation
        self.mag_transformer = Transformer(
                input_size=16, drop_rate=0.1, eps=1e-7
            )
        self.mag_lstm = nn.LSTM(16, 16, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.mag_decoder = nn.Sequential(nn.Conv1d(32, 16, kernel_size=3, padding='same'),
                                        nn.MaxPool1d(3),
                                        nn.Conv1d(16, 8, kernel_size=3, padding='same'),
                                        nn.MaxPool1d(2),
                                        nn.Conv1d(8, 4, kernel_size=2, padding='same'),
                                        nn.MaxPool1d(2),
                                        nn.Conv1d(4, 1, kernel_size=2, padding='same'),
                                        nn.MaxPool1d(2))
        
    def forward(self, x, stft):
        
        # Shared encoder part
        x = self.encoder(x)
        x = self.res_cnn_stack(x)
        x = self.bi_lstm_stack(x)
                
        seg_out = self.sigmoid(self.seg_projector(self.upconv(x).permute(0,2,1)).permute(0,2,1))
        
        # Magnitude estimation part
        tmp_mag_out, _ = self.mag_transformer(x)
        tmp_mag_out, _ = self.mag_lstm(tmp_mag_out.permute(0,2,1))
        mag_out = self.mag_decoder(tmp_mag_out.permute(0,2,1)).squeeze()

        if not self.isConformer:
            x, _ = self.transformer_d0(x)
            x, _ = self.transformer_d(x)
        else:
            x, _ = self.conformer(x.permute(0, 2, 1), 47)
            x = x.permute(0, 2, 1)
        
        # Detection part
        detection_hiddenSTFT = self.stft_cross[0](x.permute(0,2,1), stft[:, :, :16], stft[:, :, :16])

        detection = self.decoder_d(x+detection_hiddenSTFT.permute(0,2,1))
        detection = torch.sigmoid(self.conv_d(detection))
        detection = torch.squeeze(detection, dim=1)  # Remove channel dimension

        outputs = [detection]

        # Pick parts
        i = 0
        for lstm, attention, decoder, conv in zip(
            self.pick_lstms, self.pick_attentions, self.pick_decoders, self.pick_convs
        ):
            px = x.permute(
                2, 0, 1
            )  # From batch, channels, sequence to sequence, batch, channels
            px = lstm(px)[0]
            px = self.dropout(px)
            px = px.permute(
                1, 2, 0
            )  # From sequence, batch, channels to batch, channels, sequence
            pick_hidden = self.stft_cross[i+1](px.permute(0,2,1), stft[:, :, :16], stft[:, :, :16])

            px, _ = attention(px+pick_hidden.permute(0,2,1))
            px = decoder(px)
            pred = torch.sigmoid(conv(px))
            pred = torch.squeeze(pred, dim=1)  # Remove channel dimension

            outputs.append(pred)
            i += 1
            
        return seg_out, mag_out, tuple(outputs)