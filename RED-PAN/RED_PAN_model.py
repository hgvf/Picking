import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RRC(nn.Module):
    def __init__(self, stride_size, nb_filter_in, nb_filter_out, RRconv_time=3, kernel_size=7, padding='same'):
        super(RRC, self).__init__()

        self.RRconv_time = RRconv_time

        if stride_size is not None:
            self.conv = nn.Sequential(nn.Conv1d(nb_filter_in, nb_filter_out, kernel_size=kernel_size, stride=stride_size, padding=kernel_size//2),
                                     nn.BatchNorm1d(nb_filter_out, eps=0.001, momentum=0.99),
                                     nn.ReLU(),
                                     nn.Dropout(0.1))
        else:
            self.conv = nn.Sequential(nn.Conv1d(nb_filter_in, nb_filter_out, kernel_size=kernel_size, padding=padding),
                                     nn.BatchNorm1d(nb_filter_out, eps=0.001, momentum=0.99),
                                     nn.ReLU(),
                                     nn.Dropout(0.1))

        self.conv1x1 = nn.Conv1d(nb_filter_out, nb_filter_out, stride=1, padding=padding, kernel_size=kernel_size)

        self.RRC_conv = nn.Sequential(nn.Conv1d(nb_filter_out, nb_filter_out, kernel_size=kernel_size, padding=padding),
                                     nn.BatchNorm1d(nb_filter_out, eps=0.001, momentum=0.99),
                                     nn.ReLU(),
                                     nn.Dropout(0.1))

        nn.init.kaiming_uniform_(self.conv[0].weight)
        nn.init.kaiming_uniform_(self.conv1x1.weight)
        nn.init.kaiming_uniform_(self.RRC_conv[0].weight)

    def forward(self, inputs):
        u = self.conv(inputs)
        
        conv1x1 = self.conv1x1(u)
        for i in range(self.RRconv_time):
            if i == 0:
                r_u = u

            r_u = r_u + u
            r_u = self.RRC_conv(r_u)

        return r_u + conv1x1

class mtan_att_block(nn.Module):
    def __init__(self, nb_filter_in, nb_filter_out, mode, strides=5, kernel_size=7, padding='same'):
        super(mtan_att_block, self).__init__()
        
        self.mode = mode
        
        # upsampling layer for decoder
        if mode == 'up':
            self.upconv = Upconv(nb_filter_in, nb_filter_out)
        
        # downsampling layer for encoder
        if mode == 'down':
            if strides is not None:
                self.down_conv = nn.Sequential(nn.Conv1d(nb_filter_in, nb_filter_out, kernel_size=kernel_size, stride=strides, padding=kernel_size//2),
                                              nn.BatchNorm1d(nb_filter_out, eps=0.001, momentum=0.99),
                                              nn.ReLU(),
                                              nn.Dropout(0.1))
            else:
                self.down_conv = nn.Sequential(nn.Conv1d(nb_filter_in, nb_filter_out, kernel_size=kernel_size, padding=padding),
                                          nn.BatchNorm1d(nb_filter_out, eps=0.001, momentum=0.99),
                                          nn.ReLU(),
                                          nn.Dropout(0.1))
            nn.init.kaiming_uniform_(self.down_conv[0].weight)
            
        # attention layer
        attn_in = nb_filter_in * 2 if mode == 'down' else nb_filter_out * 3
        self.attn = nn.Sequential(nn.Conv1d(attn_in, nb_filter_in, kernel_size=1, stride=1, padding='valid'),
                                 nn.BatchNorm1d(nb_filter_in, eps=0.001, momentum=0.99),
                                 nn.ReLU(),
                                 nn.Conv1d(nb_filter_in, nb_filter_in, kernel_size=1, stride=1, padding='valid'),
                                 nn.BatchNorm1d(nb_filter_in, eps=0.001, momentum=0.99),
                                 nn.Sigmoid())
        
    def forward(self, pre_att_layer, pre_target, target):
        if self.mode == 'up':
            x = self.upconv(pre_att_layer, pre_target)
        else:
            # concate two tensors on input dimension
            x = torch.cat((pre_att_layer, pre_target), dim=1)
        
        # attention layer
        x = self.attn(x)

        # decoder
        if self.mode == 'up':
            mul = torch.mul(x, target)
        
        if self.mode == 'down':
            mul = torch.mul(x, target)
            mul = self.down_conv(mul)
        
        return mul

class Upconv(nn.Module):
    def __init__(self, nb_filter_in, nb_filter_out, upsize=5, padding='same', kernel_size=7):
        super(Upconv, self).__init__()

        self.conv = nn.Sequential(nn.Upsample(scale_factor=upsize),
                                 nn.Conv1d(nb_filter_in, nb_filter_out, kernel_size=kernel_size, padding=padding),
                                 nn.BatchNorm1d(nb_filter_out, eps=0.001, momentum=0.99),
                                 nn.ReLU(),
                                 nn.Dropout(0.1))

        nn.init.kaiming_uniform_(self.conv[1].weight)

    def forward(self, inputs, concatenate_layer):
        u = self.conv(inputs)

        # 長度相減
        shape_diff = u.shape[-1] - concatenate_layer.shape[-1]
        if shape_diff > 0:
            crop_shape = (shape_diff//2, shape_diff-shape_diff//2)
        else:
            crop_shape = None

        # 在 time dimension 上做 cropping
        if crop_shape:
            crop = u[:, :, crop_shape[0]:u.shape[-1]-crop_shape[1]]
        else:
            crop = u

        return torch.cat((concatenate_layer, crop), dim=1)

class RED_PAN(nn.Module):
    def __init__(self):
        super(RED_PAN, self).__init__()
        
        self.nb_filters = [6, 12, 18, 24, 30, 36]
        self.depth = 6
        self.kernel_size = 7
        self.pool_size = 5
        self.stride_size = 5
        self.upsize = 5
        self.padding = 'same'
        self.RRconv_time = 3
        
        # initialize
        self.init_RRC = RRC(None, 3, self.nb_filters[0])
        self.init_PS = mtan_att_block(nb_filter_in=self.nb_filters[0], nb_filter_out=self.nb_filters[0], mode='down', strides=None)
        self.init_M = mtan_att_block(nb_filter_in=self.nb_filters[0], nb_filter_out=self.nb_filters[0], mode='down', strides=None)
        
        # Encoder
        self.encoder_exp_RRC = nn.ModuleList([RRC(None, self.nb_filters[i], self.nb_filters[i]) for i in range(len(self.nb_filters)-1)])
        self.encoder_PS = nn.ModuleList([mtan_att_block(nb_filter_in=self.nb_filters[i], nb_filter_out=self.nb_filters[i+1], mode='down') for i in range(len(self.nb_filters)-1)])
        self.encoder_M = nn.ModuleList([mtan_att_block(nb_filter_in=self.nb_filters[i], nb_filter_out=self.nb_filters[i+1], mode='down') for i in range(len(self.nb_filters)-1)])
        self.encoder_E = nn.ModuleList([RRC(self.stride_size, self.nb_filters[i], self.nb_filters[i+1]) for i in range(len(self.nb_filters)-1)])
    
        # bottleneck layer
        self.bottleneck_RRC = RRC(None, self.nb_filters[5], self.nb_filters[5])
        
        # Decoder
        self.decoder_upconv = nn.ModuleList([Upconv(self.nb_filters[-1], self.nb_filters[-1])])
        self.decoder_upconv += nn.ModuleList([Upconv(self.nb_filters[-1-i], self.nb_filters[-2-i]) for i in range(len(self.nb_filters)-1)])
        self.decoder_RRC = nn.ModuleList([RRC(None, self.nb_filters[-1-i]*2, self.nb_filters[-1-i]) for i in range(len(self.nb_filters))])
        self.decoder_PS = nn.ModuleList([mtan_att_block(self.nb_filters[-1-i], self.nb_filters[-1-i], mode='up') for i in range(len(self.nb_filters))])
        self.decoder_M = nn.ModuleList([mtan_att_block(self.nb_filters[-1-i], self.nb_filters[-1-i], mode='up') for i in range(len(self.nb_filters))])
        
        # Output
        self.outPS = nn.Conv1d(self.nb_filters[0], 3, kernel_size=1, padding='valid')
        self.outM = nn.Conv1d(self.nb_filters[0], 2, kernel_size=1, padding='valid')
        
        nn.init.kaiming_uniform_(self.outPS.weight)
        nn.init.kaiming_uniform_(self.outM.weight)
        
    def forward(self, inputs):
        exp_Es, Es, PS_mtan_Es, M_mtan_Es = [], [], [], []

        # initialize
        conv_init_exp = self.init_RRC(inputs)
        PS_mtan_init = self.init_PS(conv_init_exp, conv_init_exp, conv_init_exp)
        M_mtan_init = self.init_M(conv_init_exp, conv_init_exp, conv_init_exp)
        
        Es.append(conv_init_exp)
        PS_mtan_Es.append(PS_mtan_init)
        M_mtan_Es.append(M_mtan_init)
        # print(f'initialize: conv_init_exp: {conv_init_exp.shape}, PS: {PS_mtan_init.shape}, M: {M_mtan_init.shape}')
        # print('='*120)
        
        # =========================================== Encoder =========================================== #
        for i in range(len(self.nb_filters)-1):
            # print(f"encoder {i}-th layer")
            if i == 0:
                exp_E = self.encoder_exp_RRC[i](conv_init_exp) # R
                # print(f"before sub-networks -> pre_att: {PS_mtan_init.shape}, pre_target: {conv_init_exp.shape}, target: {exp_E.shape}")
                PS_mtan_E = self.encoder_PS[i](PS_mtan_init, conv_init_exp, exp_E)
                M_mtan_E = self.encoder_M[i](M_mtan_init, conv_init_exp, exp_E)
                # print(f"after sub-networks -> PS/M: {PS_mtan_E.shape}")
                E = self.encoder_E[i](exp_E)
                # print(f"after sub-networks -> E: {E.shape}")
                # print('='*120)
            else:
                exp_E = self.encoder_exp_RRC[i](E) # R
                # print(f"pre_att: {PS_mtan_E.shape}, pre_target: {E.shape}, target: {exp_E.shape}")
                PS_mtan_E = self.encoder_PS[i](PS_mtan_E, E, exp_E)
                M_mtan_E = self.encoder_M[i](M_mtan_E, E, exp_E)
                # print(f"after sub-networks -> PS/M: {PS_mtan_E.shape}")
                E = self.encoder_E[i](exp_E)
                # print(f"after sub-networks -> E: {E.shape}")
                # print('='*120)
                
            PS_mtan_Es.append(PS_mtan_E)
            M_mtan_Es.append(M_mtan_E)
            Es.append(E)
            exp_Es.append(exp_E)
            
            # ============================================ bottleneck layer ============================================ #
            if i == len(self.nb_filters)-2:
                exp_E = self.bottleneck_RRC(E)
                # print(f"bottleneck layer: input: {E.shape}, output: {exp_E.shape}")
                exp_Es.append(exp_E)
                
        # =========================================== Decoder =========================================== #
        Ds, PS_mtan_Ds, M_mtan_Ds = [], [], []
        for i in range(len(self.nb_filters)):
            # print(f"decoder {i}-th layer")
            if i == 0:
                D = self.decoder_upconv[i](Es[-1], Es[-1-i])
            else:
                D = self.decoder_upconv[i](D_fus, Es[-1-i])

            D_fus = self.decoder_RRC[i](D)
            # print(f"D: {D.shape}, D_fus: {D_fus.shape}")
            
            PS_mtan_D = self.decoder_PS[i](PS_mtan_Es[-1-i], D, D_fus)
            M_mtan_D = self.decoder_M[i](M_mtan_Es[-1-i], D, D_fus)
            # print(f"after sub-networks: PS/M: {PS_mtan_D.shape}")
            # print('='*120)
            Ds.append(D_fus)
            PS_mtan_Ds.append(PS_mtan_D)
            M_mtan_Ds.append(M_mtan_D)
            
        # =========================================== Output =========================================== #
        PS_out = self.outPS(PS_mtan_D)
        M_out = self.outM(M_mtan_D)
        # print(f"Output layer: {PS_out.shape}, {M_out.shape}")
        PS_out = F.softmax(PS_out, dim=1)
        M_out = F.softmax(M_out, dim=1)
        
        return (PS_out, M_out)

