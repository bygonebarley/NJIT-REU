import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import load_dicts as ld
import pickle
import sys
from scipy import stats
from datetime import datetime

def main():
    
    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'test'

    diff = ld.load_float_diff(fm)

    normal_file = open('normal.pkl','rb')
    norm = pickle.load(normal_file)

    exponential_file = open('exponential.pkl','rb')
    expon = pickle.load(exponential_file)

    times,diffs,last = ld.load_dicts(fm)

    key = list(diff.keys())[1]
    #k,p = stats.kstest(diff[key],norm)
    #print(f'n {p = }')
    #k,p = stats.kstest(diff[key],expon)
    #print(f'e {p = }')
    #k,p = stats.kstest(norm,expon)
    #print(f'r {p = }')
    
    n = 0
    e = 0
    o = 0

    keys = []
    norm_statistical = []
    norm_p_value = []
    expo_statistical = []
    expo_p_value = []
    last_proportion = []

    for k,value in diff.items():

        deltalast = deltatime_to_float(last[k])
        np_value = np.array(value)
        sn,pn = stats.kstest((np_value-np_value.mean())/np_value.std(),'norm',alternative='two-sided')
        se,pe = stats.kstest(np_value/np_value.mean(),'expon',alternative='two-sided')
        
        keys.append(k)
        norm_statistical.append(sn)
        norm_p_value.append(pn)
        expo_statistical.append(se)
        expo_p_value.append(pe)
        last_proportion.append(prop_to_last(value,deltalast))
        
        if (k == 'C002'):
            print({deltalast})
            print(max(value))

        if (pe > 0.05):
            #print(pn)
            #n += 1
            e += 1
        elif (pn > 0.05):
            #e += 1
            n += 1
        else:
            o += 1
    
    print(f'the number of distributions that follow the normal      {n}')
    print(f'the number of distributions that follow the exponential {e}')
    print(f'the number of distributions that follow neither         {o}')
    

    df = pd.DataFrame()
    df['keys'] = keys
    df['norm_stat'] = norm_statistical
    df['norm_p'] = norm_p_value
    df['expo_stat'] = expo_statistical
    df['expo_p'] = expo_p_value
    df['last_prop'] = last_proportion

    print(df)

    #plt.figure('normal')
    #plt.hist(norm)
    #plt.figure('exponential')
    #plt.hist(expon)
    plt.figure('first key')
    plt.hist(diff[key])#,density=True)

def shift_mean(arr):
    mean = sum(arr)/len(arr)
    for i in range(len(arr)):
        if mean > 0:
            arr[i] -= mean
        else:
            arr[i] += mean
    return arr

def shift_arr(arr,val):
    for i in range(len(arr)):
        if val > 0:
            arr[i] += val
    return arr

def main2():
    
    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'Sale Data'

    diff = ld.load_float_diff(fm)

    file1 = open(fm+'_avg_mean.pkl','rb')
    avg_mean = pickle.load(file1)

    times,diffs,last = ld.load_dicts(fm)

    mor = 0
    les = 0
    most_confidence = 0

    for key,value in diff.items():
        deltalast = deltatime_to_float(last[key])
        if (prop_to_last(value,deltalast) > 0.8):
            mor += 1
            if (sum(value)/len(value) < avg_mean):
                most_confidence += 1
        else:
            les += 1

    print(f'last visit above 80% {mor}')
    print(f'last visit below 80% {les}')
    print(f'{most_confidence}')

        #print(value)
        #print(deltalast)
        #print(proportion_to_last(value,deltalast))

def deltatime_to_float(time):
    return (float(time.days) + (float(time.seconds)/(24*60*60)))

def old_prop_to_last(times,last):
    index = len(times)
    times.sort()
    for i in range(len(times)):
        if (times[i] > last):
            index = i
            break
    return (1 - float((len(times)-index)/len(times)))

def prop_to_last(times,last):
    t = np.array(times)
    return len(t[t<last])/len(t)


if __name__ == "__main__":
    start = datetime.now()
    
    main()
    
    end = datetime.now()-start
    print(f"{end.seconds} seconds")
    plt.show()
