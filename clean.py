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

    code_time = (datetime.now()-start)
    print(str(code_time.seconds) + " seconds")
