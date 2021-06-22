import load_dicts
import pandas as pd
import pickle
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import sys

if __name__ == '__main__':
    start = datetime.now()

    if (len(sys.argv) > 1):
        fm = sys.argv[1]
    else:
        fm = 'Sale Data'

    [times,diff,last] = load_dicts.load_dicts(fm)
    
    #for key in diff.keys():
        #print(diff[key])
        #for t in diff[key]:
        #    print(t.days,end=' ')
        #print(" ")

    delete = []
    for key in diff.keys():
        if (len(diff[key]) < 10):
            delete.append(key)
            continue
        if (key == 'misc'):
            delete.append('misc')

    for i in range(len(delete)):
        diff.pop(delete[i])
        times.pop(delete[i])
        last.pop(delete[i])

    diff_float = {}

    for key in diff.keys():
        d = []
        for time in diff[key]:
            time_passed = (float(time.days) + (float(time.seconds)/(24*60*60)))
            d.append(time_passed)
        diff_float[key] = d

    filename = open(fm+'_float_diff.pkl','wb')
    pickle.dump(diff_float,filename)
    filename.close()

    print(len(diff))

    p_vals = []
    d = []
    k = []
    st_cv = []
    sts = []
    means = []
    ranges = []
    ds = []
    all_days = []
    pst = []
    lst = []
    less = 0
    more = 0
    inf = 0
    norm_diff = 0
    expon_diff = 0
    other = 0

    for key in diff.keys():
        d = []
        for time in diff[key]:
            time_passed = float(time.days) + (float(time.seconds)/(24*60*60))
            d.append(time_passed)
            all_days.append(time_passed)
            #if (time.seconds > (60*60*12)):
            #    d.append(time.days+1)
            #    all_days.append(time.days+1)
            #else:
            #    d.append(time.days)
            #    all_days.append(time.days)
        #k2, p = stats.normaltest(d)
        #k2, p = stats.kstest(d,stats.norm.fit(d))
        st,cv,sl = stats.anderson(d,'expon')
        nst,ncv,nsl = stats.anderson(d,'norm')
        st_cv.append(cv[2]-st)
        sts.append(st)
        if (st>cv[2]):
            more += 1
            lst.append(len(d))
        else: 
            pst.append(len(d))
            less += 1
        #p_vals.append(p)
        #k.append(k2)
        ex = cv[2]-st
        nm = ncv[2]-nst
        #if (load_dicts.best_dist(d)=='norm'):
        #    norm_diff += 1
        #elif (load_dicts.best_dist(d)=='expon'):
        #    expon_diff += 1
        #else:
        #    other += 1
        if ((ex<0)and(nm<0)):
            other += 1
        elif (ex > nm):
            expon_diff += 1
        else:
            norm_diff += 1
        means.append(sum(d)/len(d))
        ranges.append(max(d)-min(d))
        ds.append(d)

    max_st = st_cv.index(max(st_cv))
    print(min(st_cv))
    #max_p = p_vals.index(min(p_vals))
    randi = randint(0,len(ds)-1)
    
    print(f"there are {inf} infinity values for st's")
    print(ds[randi])
    print(f"there are {more} customers that don't follow the expon - average length {sum(lst)/len(lst)}")
    print(f"there are {less} customers that follow the expon       - average length {sum(pst)/len(pst)}")
    print(f"there are {norm_diff} that resemble a normal distribution")
    print(f"there are {expon_diff} that resemble a expon distribution")
    print(f"there are {other} that resemble neither distribution")
    


    avg_len = 0

    for d in ds:
        avg_len += len(d)

    avg_len = avg_len/len(ds)

    print(f"the average length for each d in the file is {avg_len}")

    code_time = (datetime.now()-start)
    print(str(code_time.seconds) + " seconds")
    

    fig, ax = plt.subplots()

    ax.set_title("P Values")
    #ax.hist(p_vals,bins=100)
    #ax.hist(k,bins=100)
    #ax.scatter(p_vals,k)
    
    val = randi
    ax.hist(ds[val],density=True)
    x = np.linspace(min(ds[val]),max(ds[val]),100)
    l,s = stats.norm.fit(ds[val])
    ax.plot(x,stats.expon.pdf(x,scale=s))
    #ax.set_xlabel('P Values with Null Hypothesis')
    #ax.set_ylabel('Number of Random')
    

    plt.show()
