import numpy as np

def IG_Cal_Incremental(IG_old,Integ_TS,pos_old,new_pos):
    """
    Calculates the information gain incrementally, based off the information
    gain previously recorded.
    This function works by finding the segment for which the new_pos splits
    in two. It then subtracts the information gain for the whole segment and
    adds the IG for the split segments. This operation has complexity O(
    :param IG_old: The information Gain of Integ_TS with positions pos_old
    :param Integ_TS: The integral (cumulative sum) of the time series to 
                    calculate the information gain on. 
                    Representes as a numpy array of shape (number of series, time)
    :param pos_old: The positions on Integ_TS that have IG_old IG
                   Represented as a numpy aray of integer positions
    :param new_pos: The new position to add to the Integ_TS, integer
    :returns: float representing the information gain of Integ_TS over pos_old 
             and new_pos splits
    """
    Num_TS = Integ_TS.shape[0]
    Len_TS = Integ_TS.shape[1]

    # Working array to keep the new values in
    TS_dist = np.zeros(Num_TS).astype(float)

    # The positions of the segment boundaries higher and lower than the new_pos
    lower = max([-1]+ [x for x in pos_old if x <= new_pos])
    higher = min([Len_TS-1] + [x for x in pos_old if x >= new_pos])

    # Calculate the entropy of the whole old segment
    for j in range(Num_TS):
      # This operation here is meant to get the sum of all the elements between
      # lower and higher, not inclusive of higher. If lower is the very 
      # beginning of the time series (represented as lower = -1), then the 
      # cumulative sum at higher is equal to the sum between the start and higher
      if lower == -1:
          TS_dist[j] = Integ_TS[j,higher]
      else:
          TS_dist[j] = Integ_TS[j,higher] - Integ_TS[j,lower]
    old_entroy = SH_Entropy(TS_dist)

    # Use the same method to calculate the entropy of the left segment and then
    # the right
    for j in range(Num_TS):
      if lower == -1:
          TS_dist[j] = Integ_TS[j,new_pos]
      else:
          TS_dist[j] = Integ_TS[j,new_pos] - Integ_TS[j,lower]
    new_entropy_left = SH_Entropy(TS_dist)

    for j in range(Num_TS):
      TS_dist[j] = Integ_TS[j,higher] - Integ_TS[j,new_pos]

    new_entropy_right = SH_Entropy(TS_dist)

    # Then we calculate the change in weighted entropy
    weighted_right = (higher - new_pos) * new_entropy_right / Len_TS
    weighted_old = (higher - lower) * old_entroy / Len_TS
    weighted_left = (new_pos - lower) * new_entropy_left / Len_TS

    entropy_change = weighted_old - weighted_left - weighted_right

    return IG_old + entropy_change

def Clean_TS(O_Integ_TS,double):
    """
      Clean_TS does three different types of cleaning based off the values
      passed from double
      double == 0, normalise all values in the series to be between 0 and 1000 (where 1000 is the max in the channel and 0 is the min)
      double == 1, Appends the reverse of the time series to remove positive correlation
      double == 2 normalises so that 0 is still 0 but 1000 is the max of the time
      series
      We reccomend to use 1 as double under most circumstances
    """
    Integ_TS=O_Integ_TS
    if Integ_TS.ndim == 1:
        Num_TS = 1
        Len_TS = Integ_TS.shape[0]
    else:
        Num_TS = Integ_TS.shape[0]
        Len_TS = Integ_TS.shape[1]
    for i in range(Num_TS):
        minVal = min(Integ_TS[i,:])
        if double == 2:
            minVal=0
        Integ_TS[i,:] = Integ_TS[i,:]-minVal
        if double != 2:
            sumVal=sum(Integ_TS[i,:])/1000
            print(sumVal)
            Integ_TS[i,:]=Integ_TS[i,:]/sumVal
        if double == 1:
            maxVal=max(Integ_TS[i,:])
            to_append = maxVal-Integ_TS[i,:]
            sumVal=sum(to_append)/1000
            Integ_TS = np.vstack((Integ_TS,np.array(to_append/sumVal)))
    return np.cumsum(Integ_TS,axis=1)

def TopDown(Multivar_TS, k, step, double=1):
    # Pre Process the time series
    Integ_TS = Clean_TS(Multivar_TS, double)

    # get size of the input
    if Integ_TS.ndim == 1:
        Len_TS = Integ_TS.shape[0]
    else:
        Len_TS = Integ_TS.shape[1]

    # maxTT is the segments found for the maximum IG found so far
    maxTT = np.zeros(k + 1).astype(int)

    # tryTT is the working segments, that we will be trying
    tryTT = np.zeros(k + 1).astype(int)

    # IG_arr is the information gain found for k
    IG_arr = np.zeros(k + 1)
    IG_arr[0] = 0

    # p values are the second derivative of the curve at all points. Used to
    # determine the mean
    p_arr = np.zeros(k + 1)
    maxIG = 0

    # Segments k times
    for i in range(k):

        # Try for a segment in j
        for j in list(range(0, Len_TS, step)):

            # Add a new segment at point j
            tryTT[i+1] = Len_TS-1
            tryTT[i] = j

            # Does an incremental IG calculation. The incremental version of
            # this function performs much better for larger k
            IG = IG_Cal_Incremental(IG_arr[i], Integ_TS, tryTT[0:i], tryTT[i])
            if IG > maxIG:
                # Record
                maxTT = tryTT.copy()
                maxIG = IG

        # If we did not make any progress from the information gain we already had
        if maxIG == IG_arr[i]: 
            # We didn't get any information gain from this, so we should not continue
            break

        tryTT=maxTT.copy()

        IG_arr[i + 1] = maxIG
        
        # If it's possible to calculate the curvature, do so and add it to p
        if i >= 1:
            p_arr[i] = (IG_arr[i] - IG_arr[i-1]) / (IG_arr[i + 1] - IG_arr[i])
    knee = np.argmax(p_arr) +1
    return tryTT,IG_arr,knee

def SH_Entropy(x):
    x = x[(x != 0)]
    p = np.true_divide(x,np.sum(x))
    return -1 * sum(p * np.log(p))
    