import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def snr_p(z,label):
    sec=5
    secr=sec*100
    picking=np.where(label==1)[0]
    
    if len(picking)==0 or picking[0]==0 or not picking.any():
        return None
    
    # signal: picking 與後 500 samples
    picking=picking[0]
    signal=z[picking:picking+secr]
    #signal=np.power(np.sum(np.power(signal,2)),0.5)
    #signal=np.std(signal)
    
    signal=np.mean(np.power(signal,2))
    
    # noise: picking 前 500 samples
    if picking<secr:
        noise=z[:picking].tolist()
        while len(noise)<secr:
            noise.extend(noise)
        noise=noise[:secr]
    else:
        noise=z[picking-secr:picking]

    #noise=np.power(np.sum(np.power(noise,2)),0.5)
    #noise=np.std(noise)
    noise=np.mean(np.power(noise,2))
    ratio=signal/noise
    
    return ratio

def plot(hist):
    snr_=np.log10(hist)

    bar_width = 1
    opacity = 0.6
    print(np.max(snr_))

    plt.figure()
    #rect1 = plt.bar(index,snr_range,bar_width,alpha = opacity,hatch="+",color = "white",edgecolor='black')
    rect1 = plt.hist(snr_,9,range=(-1,8),rwidth=bar_width,alpha = opacity,edgecolor='black')
    #plt.hist(snr_,alpha = opacity,edgecolor='black')
    #plt.xticks(map(lambda x: x-bar_width/2.0,np.arange(6)))

    plt.ylabel('Number of samples')
    plt.xlabel('Log(SNR)')
    #plt.show()
    plt.savefig('original_training_data.png')

if __name__ == '__main__':
    x = np.load('x.npy', allow_pickle=True)
    y = np.load('y_p.npy', allow_pickle=True)

    hist = []
    for i in tqdm(range(x.shape[0])):
        hist += [snr_p(x[i, :, -1], y[i])]

    plot(hist)

    # x = np.load('x_new.npy', allow_pickle=True)
    # y = np.load('y_new_p.npy', allow_pickle=True)

    # data = []
    # for i in range(x.shape[0]):
    #     if y[i].any():
    #         data.append(np.hstack((x[i], y[i])))


    # data = np.array(data, dtype=object)
    # data = data.astype(float)

    # hist = []
    # for i in tqdm(range(data.shape[0])):
    #     hist += [snr_p(data[i, :, -2], data[i, :, -1])]
    
    # new_hist = []
    # for i in range(len(hist)):
    #     if hist[i] is not None:
    #         new_hist += [hist[i]]
    
    # plot(new_hist)    