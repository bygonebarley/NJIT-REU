import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import load_dicts as ld
import pickle
import sys
from scipy import stats
from datetime import datetime
import math

def main():
    
    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'test'

    diff = ld.load_float_diff(fm)
    times,diffs,last = ld.load_dicts(fm)


    df = pd.read_pickle(fm+"_target_att.pkl")

    # Columns - 'keys' 'norm_stat_nf' 'norm_p_nf' 'norm_stat_ks' 'norm_p_ks' 
    #           'expo_stat' 'expo_p' 'last_prop'

    ks = 0
    nf = 0
    norm = 0
    expo = 0
    other = 0
    dist = []

    for index,row in df.iterrows():
        #if (row['norm_p_nf'] > row['norm_p_ks']):
        #    nf  += 1
        #else:
        #    ks += 1
        if (row['expo_p'] > 0.0001):
            expo += 1
            dist.append('e')
        else:
            other += 1
            dist.append('o')

    
    print(f'kstest p-values {ks}')
    print(f'normaltest p-values {nf}')
    df['dist'] = dist
    #print(df)
    
    # stayed/left 0-1, the lower the number the greater chance they stayed 
    stayed = []
    mean_time = []
    max_time = []
    last_time = []
    has_stayed = []

    for index,row in df.iterrows():
        np_diff = np.array(diff[row['keys']])  
        deltalast = deltatime_to_float(last[row['keys']])
        prob = stats.expon.cdf(deltalast,scale=np_diff.mean())
        stayed.append(prob)
        mean_time.append(np_diff.mean())
        max_time.append(max(np_diff))
        last_time.append(deltalast)

        if prob > 0.99:
            has_stayed.append(1)
        elif prob < 0.5:
            has_stayed.append(0)
        else:
            has_stayed.append(float('nan'))

        if (row['dist'] == 'e'):
            probability = 1 - np.exp((-1*deltalast)/np_diff.mean())
            prob = stats.expon.cdf(deltalast,scale=np_diff.mean())            
            if index < 10:
                print(f"{row['last_prop']:8.6f} ",end=' ')
                print(f"{probability:8.6f} ",end=' ')
                print(f"{prob:8.6f}")
            #stayed.append(probability)
        elif (row['dist'] == 'n'):
            #print(f"{row['last_prop'] = }")          
            probability =  0.5 + (math.erf( (deltalast - np_diff.mean())/(np_diff.std()*math.sqrt(2))  ))/2
            #print(f"{probability = }")
            #stayed.append(probability)
        else:
            continue
            #stayed.append(row['last_prop'])
       
    stay_np = np.array(stayed)
    df['cdf'] = stayed
    df['mean'] = mean_time
    df['max'] = max_time
    df['last'] = last_time
    #print(df.sort_values(by=['cdf']))
    print(len(stay_np[stay_np<0.5]),end=' ')
    print(' stayed')
    print(len(stay_np[stay_np>0.5]),end=' ')
    print(' left')

    odf = pd.DataFrame()
    odf['keys'] = df['keys']
    odf['cdf'] = df['cdf']
    odf['mean'] = df['mean']
    odf['max'] = df['max']
    odf['last'] = df['last']
    odf['stayed'] = has_stayed
    print(odf)
    odf.to_pickle(fm+'_churn.pkl')
    plt.figure('AMOHData')
    plt.title('AMOHData')
    plt.hist(stay_np)

def main2():
    
    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'Sale Data'

    diff = ld.load_float_diff(fm)
    times,diffs,last = ld.load_dicts(fm)

    df = pd.read_pickle(fm+"_target_att.pkl")

    # Columns - 'keys' 'norm_stat_nf' 'norm_p_nf' 'norm_stat_ks' 'norm_p_ks' 
    #           'expo_stat' 'expo_p' 'last_prop'

    ks = 0
    nf = 0
    norm = 0
    expo = 0
    other = 0
    dist = []

    for index,row in df.iterrows():
        #if (row['norm_p_nf'] > row['norm_p_ks']):
        #    nf  += 1
        #else:
        #    ks += 1
        norm_p = (row['norm_p_nf']+row['norm_p_ks'])/2
        if (norm_p > row['expo_p']):
            if (norm_p > 0.05):
                norm += 1
                dist.append('n')
            else:
                other += 1
                dist.append('o')
        else:
            if (row['expo_p'] > 0.05):
                expo += 1
                dist.append('e')
            else:
                other += 1
                dist.append('o')

    
    #print(f'kstest p-values {ks}')
    #print(f'normaltest p-values {nf}')
    df['dist'] = dist
    print(df)

    print(f'norm p values {norm}')
    print(f'expo p values {expo}')
    print(f'other  values {other}')
    
def main3():
    
    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'Sale Data'

    diff = ld.load_float_diff(fm)
    times,diffs,last = ld.load_dicts(fm)

    df = pd.read_pickle(fm+"_target_att.pkl")

    # Columns - 'keys' 'norm_stat_nf' 'norm_p_nf' 'norm_stat_ks' 'norm_p_ks' 
    #           'expo_stat' 'expo_p' 'last_prop'

    ks = 0
    nf = 0
    norm = 0
    expo = 0
    other = 0
    dist = []

    for index,row in df.iterrows():
        if (row['norm_p_nf'] > row['norm_p_ks']):
            nf  += 1
        else:
            ks += 1
    
    print(df)
    print(f'kstest p-values {ks}')
    print(f'normaltest p-values {nf}')
    
def deltatime_to_float(time):
    return (float(time.days) + (float(time.seconds)/(24*60*60)))

if __name__ == "__main__":
    start = datetime.now()

    main()

    end = datetime.now()-start
    print(f"{end.seconds} seconds")
    plt.show()
