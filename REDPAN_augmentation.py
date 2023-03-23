import torch
import numpy as np
import random

def MMWA_Taiwan(data, gt):
    # (batch, 8, 3000)
    batch = data.shape[0]

    ptime = [torch.argmax(gt[i, 0]) for i in range(batch)]
    seis = [True if p != 0 else False for p in ptime]
    seis_idx = np.arange(batch)[seis]

    marching_idx = {}
    for i in range(batch):
        if ptime[i] == 0.0:
            continue
        marching_idx[i] = seis_idx[random.sample(range(seis.count(True)), 1)[0]]

    newdata = data.clone()
    newgt = gt.clone()
    for i in range(batch):
        try:
            # noise
            if ptime[i] == 0.0:
                newdata[i] = data[i]
                continue

            # 兩個事件中間的間隔
            gap = np.random.randint(low=300, high=500)

            # first & second event: (before p, after p)
            if ptime[i] > 500:
                first_event = [np.random.randint(low=100, high=500), np.random.randint(low=500, high=1000)]
            else:
                first_event = [np.random.randint(low=ptime[i]-30, high=ptime[i]-10), np.random.randint(low=500, high=1000)]

            if ptime[marching_idx[i]] > 500:
                second_event = [np.random.randint(low=100, high=500), np.random.randint(low=500, high=1000)]
            else:
                second_event = [np.random.randint(low=ptime[marching_idx[i]]-30, high=ptime[marching_idx[i]]-10), np.random.randint(low=500, high=1000)]

            wave_length = first_event[-1]+first_event[0] + second_event[-1]+second_event[0] + gap
            remaining_wave = 3000 - wave_length

            if wave_length > 3000 or wave_length < 0:
                continue

            toConcat = torch.cat((data[i, :, ptime[i]-first_event[0]:ptime[i]+first_event[-1]], torch.zeros((12, gap)),
                                    data[marching_idx[i], :, ptime[marching_idx[i]]-second_event[0]:ptime[marching_idx[i]]+second_event[-1]], torch.zeros((12, remaining_wave))), dim=-1)

            GTtoConcat0 = torch.cat((gt[i, 0, ptime[i]-first_event[0]:ptime[i]+first_event[-1]], torch.zeros((gap)),
                                    gt[marching_idx[i], 0, ptime[marching_idx[i]]-second_event[0]:ptime[marching_idx[i]]+second_event[-1]], torch.zeros((remaining_wave))), dim=-1)

            GTtoConcat1 = torch.cat((gt[i, 1, ptime[i]-first_event[0]:ptime[i]+first_event[-1]], torch.ones((gap)),
                                    gt[marching_idx[i], 1, ptime[marching_idx[i]]-second_event[0]:ptime[marching_idx[i]]+second_event[-1]], torch.ones((remaining_wave))), dim=-1)

            if toConcat.shape[-1] != 3000:
                newdata[i] = torch.cat((toConcat, torch.zeros(12, 3000-toConcat.shape[-1])))
            else:
                newdata[i] = toConcat

            if GTtoConcat0.shape[-1] != 3000:
                newgt[i, 0] = torch.cat((GTtoConcat0, torch.zeros(3000-GTtoConcat0.shape[-1])))
            else:
                newgt[i, 0] = GTtoConcat0

            if GTtoConcat1.shape[-1] != 3000:
                newgt[i, 1] = torch.cat((GTtoConcat1, torch.ones(3000-GTtoConcat1.shape[-1])))
            else:
                newgt[i, 1] = GTtoConcat1

        except Exception as e:
            # print(e)
            newdata[i] = data[i]
            newgt[i] = gt[i]
    return newdata, newgt

def EEWA_Taiwan(data, gt):
    # (batch, 8, 3000)
    batch = data.shape[0]

    ptime = [torch.argmax(gt[i, 0]) for i in range(batch)]
    seis = [True if p != 0 else False for p in ptime]
    seis_idx = np.arange(batch)[seis]

    n_marching = {}
    marching_idx = {}
    for i in range(batch):
        if ptime[i] == 0.0:
            continue
        n_marching[i] = np.random.randint(low=2, high=4)
        marching_idx[i] = seis_idx[random.sample(range(seis.count(True)), n_marching[i])]

    newdata = data.clone()
    newgt = gt.clone()
    for i in range(batch):
        try:
            # noise
            if ptime[i] == 0.0:
                newdata[i] = data[i]
                continue
            toMarching_idx = marching_idx[i]

            # 兩個事件中間的間隔
            gap = [np.random.randint(low=300, high=500) for _ in range(n_marching[i])]

            wave_length = []
            # first event's length
            if ptime[i] > 350:
                event = [np.random.randint(low=200, high=350), np.random.randint(low=50, high=300)]
            else:
                event = [np.random.randint(low=ptime[i]-30, high=ptime[i]-10), np.random.randint(low=50, high=300)]
            wave_length.append(event)

            for march in marching_idx[i]:
                if ptime[march] > 350:
                    event = [np.random.randint(low=200, high=350), np.random.randint(low=50, high=300)]
                else:
                    event = [np.random.randint(low=ptime[march]-30, high=ptime[march]-10), np.random.randint(low=50, high=300)]

                wave_length.append(event)

            length = np.sum(wave_length) + np.sum(gap)
            remaining_wave = 3000 - length

            if length > 3000 or length < 0:
                continue

            toConcat = torch.empty((12, 3000))
            GTtoConcat0, GTtoConcat1 = torch.empty(3000), torch.empty(3000)
            for idx, event_idx in enumerate(wave_length):
                if idx == 0:
                    toConcat = data[i, :, ptime[i]-event_idx[0]:ptime[i]+event_idx[1]]
                    GTtoConcat0 = gt[i, 0, ptime[i]-event_idx[0]:ptime[i]+event_idx[1]]
                    GTtoConcat1 = gt[i, 1, ptime[i]-event_idx[0]:ptime[i]+event_idx[1]]
                else:
                    toConcat = torch.cat((toConcat, torch.zeros((12, gap[idx-1])), data[toMarching_idx[idx-1], :, ptime[toMarching_idx[idx-1]]-event_idx[0]:ptime[toMarching_idx[idx-1]]+event_idx[1]]), dim=-1)
                    GTtoConcat0 = torch.cat((GTtoConcat0, torch.zeros(gap[idx-1]), gt[toMarching_idx[idx-1], 0, ptime[toMarching_idx[idx-1]]-event_idx[0]:ptime[toMarching_idx[idx-1]]+event_idx[1]]), dim=-1)
                    GTtoConcat1 = torch.cat((GTtoConcat1, torch.ones(gap[idx-1]), gt[toMarching_idx[idx-1], 1, ptime[toMarching_idx[idx-1]]-event_idx[0]:ptime[toMarching_idx[idx-1]]+event_idx[1]]), dim=-1)

            toConcat = torch.cat((toConcat, torch.zeros((12, remaining_wave))), dim=-1)
            GTtoConcat0 = torch.cat((GTtoConcat0, torch.zeros(remaining_wave)), dim=-1)
            GTtoConcat1 = torch.cat((GTtoConcat1, torch.ones(remaining_wave)), dim=-1)

            newdata[i] = toConcat
            newgt[i, 0] = GTtoConcat0
            newgt[i, 1] = GTtoConcat1
        except Exception as e:
            # print(e)
            newdata[i] = data[i]
            newgt[i] = gt[i]
    return newdata, newgt

def REDPAN_augmentation(data, gt):
    prob = np.random.uniform()

    try:
        if prob < 0.3:
            data, gt = MMWA_Taiwan(data, gt)
        elif prob >= 0.3 and prob < 0.6:
            data, gt = EEWA_Taiwan(data, gt)
    except Exception as e:
        # print(e)
        pass

    return data, gt
