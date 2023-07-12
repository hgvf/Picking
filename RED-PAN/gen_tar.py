import numpy as np
import torch
import random

def gen_tar_func(data_length, point, mask_window):
    '''
    data_length: target function length
    point: point of phase arrival
    mask_window: length of mask, must be even number
                 (mask_window//2+1+mask_window//2)
    '''
    target = np.zeros(data_length)
    half_win = mask_window//2
    gaus = np.exp(-(
        np.arange(-half_win, half_win+1))**2 / (2*(half_win//2)**2))
    #print(gaus.std())
    gaus_first_half = gaus[:mask_window//2]
    gaus_second_half = gaus[mask_window//2+1:]
    target[point] = gaus.max()
    #print(gaus.max())
    if point < half_win:
        reduce_pts = half_win-point
        start_pt = 0
        gaus_first_half = gaus_first_half[reduce_pts:]
    else:
        start_pt = point-half_win
    target[start_pt:point] = gaus_first_half
    target[point+1:point+half_win+1] = \
        gaus_second_half[:len(target[point+1:point+half_win+1])]

    return target

def gen_target(ptime=None, stime=None, p_win=40, s_win=60, length=3000):
    # print(f"ptime: {ptime}, stime: {stime}")
    trc_tn = np.ones(length)
    trc_mask = 0
    if ptime is not None:
        trc_tp = gen_tar_func(length, ptime, p_win)
        trc_tn -= trc_tp
        trc_mask += trc_tp
    if stime is not None:
        if stime == 3000:
            stime = 2999
            
        trc_ts = gen_tar_func(length, stime, s_win)
        trc_tn -= trc_ts
        trc_mask += trc_ts
    else:
        stime = ptime + np.random.randint(low=250, high=600)
        trc_ts = np.zeros(length)

    trc_mask[ptime:stime+1] = 1
    trc_unmask = np.ones(length) - trc_mask

    return np.array([trc_tp, trc_ts, trc_tn]), np.array([trc_mask, trc_unmask])

def DWA(task1_loss, task2_loss, cur_epoch, cur_PS_loss, cur_M_loss):
    H = torch.FloatTensor([2])
    K = torch.FloatTensor([2, 2])
    
    # cur_epoch 從 0 開始
    if cur_epoch <= 1:
        w1, w2 = 1, 1
    else:
        w1 = task1_loss[cur_epoch-1] / task1_loss[cur_epoch-2]      # w_pick
        w2 = task2_loss[cur_epoch-1] / task2_loss[cur_epoch-2]      # w_mask

    w = torch.FloatTensor([w1, w2])
    w = torch.exp(w/H)
    
    denom = torch.sum(w)
    lambda_weight = K * w / denom 
    
    return lambda_weight[0]*cur_PS_loss + lambda_weight[1]*cur_M_loss
    # return torch.sum(lambda_weight*torch.FloatTensor([cur_PS_loss, cur_M_loss]))

def MMWA(data):
    # (batch, 8, 3000)
    batch = data.shape[0]

    marching_idx = [random.sample(range(batch), 1)[0] for _ in range(batch)]
    ptime = [torch.argmax(data[i, 3])-100 for i in range(batch)]
    stime = [torch.argmax(data[i, 4]) for i in range(batch)]

    newdata = data.clone()
    for i in range(batch):
        try:
            # noise
            if ptime == -100:
                newdata[i] = data[i]

            # 兩個事件中間的間隔
            gap = np.random.randint(low=300, high=500)
            wave_length = (stime[i]+300-ptime[i]) + (stime[marching_idx[i]]+300-ptime[marching_idx[i]]) + gap
            remaining_wave = 3000 - wave_length
            init = np.random.randint(low=100, high=min(300, remaining_wave))
            if wave_length > 3000:
                continue

            remaining = 3000 - init - (stime[i]+300-ptime[i]) - (stime[marching_idx[i]]+300-ptime[marching_idx[i]])
            out = torch.cat((torch.ones(3, init)*data[i, :3, 10][:, None], data[i, :3, ptime[i]:stime[i]+300], torch.ones(3, max(remaining//2, 300))*data[i, :3, 10][:, None], data[marching_idx[i], :3, ptime[marching_idx[i]]:stime[marching_idx[i]]+300]), dim=-1)
            psn_out = torch.cat((torch.zeros(2, init), data[i, 3:5, ptime[i]:stime[i]+300], torch.zeros(2, max(remaining//2, 300)), data[marching_idx[i], 3:5, ptime[marching_idx[i]]:stime[marching_idx[i]]+300]), dim=-1)
            psn_out_last = torch.cat((torch.ones(init), data[i, 5, ptime[i]:stime[i]+300], torch.ones(max(remaining//2, 300)), data[marching_idx[i], 5, ptime[marching_idx[i]]:stime[marching_idx[i]]+300]), dim=-1)
            mask_out = torch.cat((torch.zeros(init), data[i, 6, ptime[i]:stime[i]+300], torch.zeros(max(remaining//2, 300)), data[marching_idx[i], 6, ptime[marching_idx[i]]:stime[marching_idx[i]]+300]), dim=-1)
            mask_out_last = torch.cat((torch.ones(init), data[i, -1, ptime[i]:stime[i]+300], torch.ones(max(remaining//2, 300)), data[marching_idx[i], -1, ptime[marching_idx[i]]:stime[marching_idx[i]]+300]), dim=-1)
            
            newdata[i, :3] = torch.cat((out, torch.ones(3, 3000-out.shape[-1])*data[i, :3, 10][:, None]), dim=-1)
            newdata[i, 3:5] = torch.cat((psn_out, torch.zeros(2, 3000-out.shape[-1])), dim=-1)
            newdata[i, 5] = torch.cat((psn_out_last, torch.ones(3000-out.shape[-1])), dim=-1)
            newdata[i, 6] = torch.cat((mask_out, torch.zeros(3000-out.shape[-1])), dim=-1)
            newdata[i, -1] = torch.cat((mask_out_last, torch.ones(3000-out.shape[-1])), dim=-1)
        except Exception as e:
            # print('MMWA: ', e)
            newdata[i] = data[i]

    return newdata

def EEWA(data):
    # (batch, 8, 3000)
    batch = data.shape[0]
    
    n_marching = [np.random.randint(low=2, high=4) for _ in range(batch)]
    marching_idx = [random.sample(range(batch), n_marching[i]) for i in range(batch)]
    ptime = [torch.argmax(data[i, 3]) for i in range(batch)]
    stime = [torch.argmax(data[i, 4]) for i in range(batch)]
 
    newdata = data.clone()
    for i in range(batch):
        try:
            # noise
            if ptime == 0:
                newdata[i] = data[i]

            gap = [np.random.randint(low=300, high=500) for _ in range(n_marching[i]-1)]
            
            wave_length = [np.random.randint(low=100, high=stime[marching_idx[i][j]]-ptime[marching_idx[i][j]]-50) for j in range(n_marching[i])]
            
            wave_length.append(np.random.randint(low=100, high=stime[i]-ptime[i]-50))
            length = sum(wave_length) + sum(gap)
            remaining_wave = 3000 - length
            init = np.random.randint(low=100, high=min(300, remaining_wave))
            if length > 3000:
                continue
            
            out = torch.cat((torch.ones(3, init)*data[i, :3, 10][:, None], data[i, :3, ptime[i]:ptime[i]+wave_length[-1]]), dim=-1)
            psn_out = torch.cat((torch.zeros(2, init), data[i, 3:5, ptime[i]:ptime[i]+wave_length[-1]]), dim=-1)
            psn_out_last = torch.cat((torch.ones(init), data[i, 5, ptime[i]:ptime[i]+wave_length[-1]]), dim=-1)
            mask_out = torch.cat((torch.zeros(init), data[i, 6, ptime[i]:ptime[i]+wave_length[-1]]), dim=-1)
            mask_out_last = torch.cat((torch.ones(init), data[i, -1, ptime[i]:ptime[i]+wave_length[-1]]), dim=-1)
            
            for j in range(len(gap)):
                out = torch.cat((out, torch.ones(3, gap[j])*data[marching_idx[i][j], :3, 10][:, None], data[marching_idx[i][j], :3, ptime[marching_idx[i][j]]:ptime[marching_idx[i][j]]+wave_length[j]]), dim=-1)
                psn_out = torch.cat((psn_out, torch.zeros(2, gap[j]), data[marching_idx[i][j], 3:5, ptime[marching_idx[i][j]]:ptime[marching_idx[i][j]]+wave_length[j]]), dim=-1)
                psn_out_last = torch.cat((psn_out_last, torch.ones(gap[j]), data[marching_idx[i][j], 5, ptime[marching_idx[i][j]]:ptime[marching_idx[i][j]]+wave_length[j]]), dim=-1)
                mask_out = torch.cat((mask_out, torch.zeros(gap[j]), data[marching_idx[i][j], 6, ptime[marching_idx[i][j]]:ptime[marching_idx[i][j]]+wave_length[j]]), dim=-1)
                mask_out_last = torch.cat((mask_out_last, torch.ones(gap[j]), data[marching_idx[i][j], -1, ptime[marching_idx[i][j]]:ptime[marching_idx[i][j]]+wave_length[j]]), dim=-1)

            newdata[i, :3] = torch.cat((out, torch.ones(3, 3000-out.shape[-1])*data[i, :3, 10][:, None]), dim=-1)
            newdata[i, 3:5] = torch.cat((psn_out, torch.zeros(2, 3000-out.shape[-1])), dim=-1)
            newdata[i, 5] = torch.cat((psn_out_last, torch.ones(3000-out.shape[-1])), dim=-1)
            newdata[i, 6] = torch.cat((mask_out, torch.zeros(3000-out.shape[-1])), dim=-1)
            newdata[i, -1] = torch.cat((mask_out_last, torch.ones(3000-out.shape[-1])), dim=-1)
            
        except Exception as e:
            # print('EEWA: ', e)
            newdata[i]= data[i]

    return newdata

def REDPAN_aug(data):
    prob = np.random.uniform()
    
    if prob < 0.24:
        data = MMWA(data)

    elif prob >= 0.24 and prob < 0.32:
        data = EEWA(data)

    return data