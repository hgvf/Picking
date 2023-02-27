import numpy as np

def Clean(wave):
    Integ_TS=wave
    # Num_TS: channel, Len_TS: length
    if Integ_TS.ndim == 1:
        Num_TS = 1
        Len_TS = Integ_TS.shape[0]
    else:
        Num_TS = Integ_TS.shape[0]
        Len_TS = Integ_TS.shape[1]
    
    minVal = np.min(Integ_TS, axis=1, keepdims=True)[0]
    Integ_TS = Integ_TS-minVal
    
    sumVal = np.sum(Integ_TS, axis=1, keepdims=True) / 1000
    Integ_TS = Integ_TS / sumVal
    
    maxVal = np.max(Integ_TS, axis=1, keepdims=True)[0]
    to_append = maxVal - Integ_TS
    sumVal = np.sum(to_append, axis=1, keepdims=True) / 1000
    to_append = to_append / sumVal
    
    Integ_TS = np.vstack((Integ_TS, to_append))
    
    return np.cumsum(Integ_TS, axis=1)

def TopDown(wave, k, step):
    Integ_TS = Clean(wave)

    # get size of the input
    if Integ_TS.ndim == 1:
        Len_TS = Integ_TS.shape[0]
    else:
        Len_TS = Integ_TS.shape[1]

    # maxTT is the segments found for the maximum IG found so far
    maxTT = np.zeros(k + 1).astype(np.int32)

    # tryTT is the working segments, that we will be trying
    tryTT = np.zeros(k + 1).astype(int)
    tryTT_j = np.zeros((k + 1, Len_TS)).astype(np.int32)

    # IG_arr is the information gain found for k
    IG_arr = np.zeros(k + 1)
    IG_arr[0] = 0
    
    # p values are the second derivative of the curve at all points. Used to
    # determine the mean
    p_arr = np.zeros(k + 1)
    maxIG = 0
    
    idx = np.arange(Len_TS)
    # Segments k times
    for i in range(k):
        tryTT[i+1] = Len_TS-1
        
        new_pos = np.arange(Len_TS)
        lower = np.ones(Len_TS) * -1
        higher = np.ones(Len_TS) * Len_TS-1

        if i != 0:
            tmp_lower = np.ones(Len_TS) * -1
            tmp_higher = np.ones(Len_TS) * Len_TS-1

            for l in tryTT[0:i]:
                lower_mask = new_pos >= l
                tmp_lower[lower_mask] = l
                lower = np.maximum(tmp_lower, lower)

                higher_mask = new_pos <= l
                tmp_higher[higher_mask] = l
                higher = np.minimum(tmp_higher, higher)

        IG = IG_Cal_Incremental(IG_arr[i], Integ_TS, lower, higher)
        IG[np.isnan(IG)] = 0

        maxIG_idx = np.argmax(IG)
        maxIG = IG[maxIG_idx]

        if maxIG == IG_arr[i]:
            break
            
        tryTT[i] = maxIG_idx
        IG_arr[i+1] = maxIG
        
        if i >= 1:
            p_arr[i] = (IG_arr[i] - IG_arr[i-1]) / (IG_arr[i + 1] - IG_arr[i])
            
    knee = np.argmax(p_arr) + 1
    return tryTT,IG_arr,knee

def SH_Entropy(x):
    p = np.divide(x, np.sum(x, axis=0))

    return -1 * np.sum(p * np.log(p), axis=0)

def IG_Cal_Incremental(IG_old, Integ_TS, lower, higher):
    # Working array to keep the new values in
    Len_TS = Integ_TS.shape[1]
    
    # TS_dist : [channel*2, timesteps]
    TS_dist = np.zeros(Integ_TS.shape).astype(np.float64)
    new_pos = np.arange(Integ_TS.shape[1])
    
    # Calculate the entropy of the whole old segment
    mask = (lower == -1)
    
    TS_dist[:, mask] = Integ_TS[:, higher[mask].astype(np.int32)]
    TS_dist[:, ~mask] = Integ_TS[:, higher[~mask].astype(np.int32)] - Integ_TS[:, lower[~mask].astype(np.int32)]
    old_entroy = SH_Entropy(TS_dist)

    # Use the same method to calculate the entropy of the left segment and then
    # the right
    TS_dist[:, mask] = Integ_TS[:, new_pos[mask].astype(np.int32)]
    TS_dist[:, ~mask] = Integ_TS[:, new_pos[~mask].astype(np.int32)] - Integ_TS[:, lower[~mask].astype(np.int32)]
    new_entropy_left = SH_Entropy(TS_dist)

    TS_dist = Integ_TS[:, higher.astype(np.int32)] - Integ_TS[:, new_pos.astype(np.int32)]
    new_entropy_right = SH_Entropy(TS_dist)

    # Then we calculate the change in weighted entropy
    weighted_right = (higher - new_pos) * new_entropy_right / Len_TS
    weighted_old = (higher - lower) * old_entroy / Len_TS
    weighted_left = (new_pos - lower) * new_entropy_left / Len_TS
    
    entropy_change = weighted_old - weighted_left - weighted_right
    return IG_old + entropy_change

