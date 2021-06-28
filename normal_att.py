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
        fm = 'AMOHData'    

    df = pd.read_pickle(fm+'_churn.pkl')
    big_df = pd.read_pickle(fm+'.pkl')
    diff = ld.load_float_diff(fm)
    all_dt,diff_dt,last_dt = ld.load_dicts(fm)

    keys = list(diff.keys())
    first = keys[0]
    print(big_df)
    
    #total = transaction_total(all_dt,keys)
    #df['total'] = total
    
    #monthly = transaction_monthly(all_dt,diff,keys)
    #df['monthly average'] = monthly
    
    #min_times = min_time_between(diff,keys)
    #df['min'] = min_times

    #total_prod = total_product(big_df,keys)
    #df['total product'] = total_prod

    #avg_prod = average_product(total,total_prod)
    #df['average product'] = avg_prod

    #min_prod,max_prod = min_max_product(big_df,keys)
    #df['minimum product'] = min_prod
    #df['maximum product'] = max_prod

    print(df)
    df.to_pickle(fm+'_churn.pkl')

def transaction_total(all,keys):
    # this function will return the total number of transactions
    arr = []
    for i in keys:
        arr.append(len(all[i]))
    return arr

def transaction_monthly(all,diff,keys):
    # this function will return the monthly average transactions
    arr =[]
    for i in keys:
        arr.append(30.416*len(all[i])/sum(diff[i]))
    return arr

def avg_time_between(diff,keys):
    # this function will return the average time in between transactions
    arr = []
    for i in keys:
        arr.append(sum(diff[i])/len(diff[i]))
    return arr

def max_time_between(diff,keys):
    # this function will return the max time in between transactions
    arr = []
    for i in keys:
        arr.append(max(diff[i]))
    return arr

def min_time_between(diff,keys):
    # this function will return the min time in between transactions
    arr = []
    for i in keys:
        arr.append(min(diff[i]))
    return arr

def total_product(df,keys):
    # this function will go through the filename.pkl dataframe and count the number of products for each customer account
    # this will not count the qty, but it will only count the total separate items
    # so 10 Breads will count the same as 1 Bread
    arr = np.zeros(len(keys))
    for index,row in df.iterrows():
        if row['Customer Account'] not in keys:
            continue
        ind = keys.index(row['Customer Account'])
        arr[ind] += 1
    return arr

def average_product(total_trans,total_prod):
    # this function will take previously generated arrays of total transactions and total purchases to 
    # calculate the average number of products purchased at each transaction
    arr = []
    for i in range(len(total_prod)):
        arr.append(total_prod[i]/total_trans[i])
    return arr

def min_max_product(df,keys):
    # this function will find the min and max number of items bought in one transaction
    dic = {}
    for i in keys:
        dic[i] = {}
    
    for index,row in df.iterrows():
        TN = row['Transaction Number']
        CA = row['Customer Account']
        if CA not in keys:
            continue
        if TN not in dic[CA]:
            dic[CA][TN] = 1
        else:
            dic[CA][TN] += 1
    min_arr = []
    max_arr = []
    for i in keys:
        temp = list(dic[i].values())
        min_arr.append(min(temp))
        max_arr.append(max(temp))
    
    return min_arr,max_arr
        
def min_max_amount(df,keys):
    # this function will find the min and max amount of money spent in one transaction
    dic = {}
    for i in keys:
        dic[i] = {}
    
    for index,row in df.iterrows():
        TN = row['Transaction Number']
        CA = row['Customer Account']
        SP = row['Sale Price']
        if CA not in keys:
            continue
        if TN not in dic[CA]:
            dic[CA][TN] = SP
        else:
            dic[CA][TN] += SP
    min_arr = []
    max_arr = []
    for i in keys:
        temp = list(dic[i].values())
        min_arr.append(min(temp))
        max_arr.append(max(temp))
    
    return min_arr,max_arr        


if __name__ == "__main__":
    start = datetime.now()

    main()

    end = datetime.now()-start
    print(f"{end.seconds} seconds")
