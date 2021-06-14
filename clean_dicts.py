import load_dicts
import pandas as pd
import pickle
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == '__main__':
    start = datetime.now()

    [a,b,c] = load_dicts.load_dicts('SD')
    
    shortened_dict = {}
    for key in b.keys():
        arr = []
        arr = b[key]
        #for diff in b[key]:
            #if (diff.seconds > 5):
                #arr.append(diff)
        if (len(arr) > 10):
            if (len(shortened_dict) < 10):
                print(key)
            shortened_dict[key] = arr

    print(len(shortened_dict))
    print(len(b))

    fi = list(shortened_dict.keys())[15]
    
    num = 0

    nums = []

    for i in shortened_dict[fi]:
        nums.append(int(i.days))
        if i.seconds < 5:
            num += 1

    print("there are " + str(num) + " times which are less than 10 seconds")
    
    print('the length of the first item in the shortened dictionary: ' + str(len(shortened_dict[fi])))

    #nums.sort()

    k2, p = stats.normaltest(nums)

    for i in range(15):
        print(a[fi][i],end="      ")
        print(b[fi][i])
        
    

    print(nums[:5])

    print("p = {:g}".format(p))

    code_time = (datetime.now()-start)
    print(str(code_time.seconds) + " seconds")
    

    #plt.scatter(range(len(nums)),nums)
    plt.hist(nums,bins=10)
    plt.show()
