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
    #diff = ld.load_float_diff(fm)
    #all_dt,diff_dt,last_dt = ld.load_dicts(fm)

    keys = list(df['Customer Account'])
    #keys = list(df['keys'])
    first = keys[0]
    #print(keys)
    #print(big_df)
    #print(df)

    #total = transaction_total(all_dt,keys)
    #df['total'] = total
    
    #monthly = transaction_monthly(all_dt,diff,keys)
    #df['monthly average'] = monthly
    
    #min_times = min_time_between(diff,keys)
    #df['min'] = min_times

    #total_prod = total_product(big_df,keys)
    #df['total product'] = total_prod

    #avg_prod = average_product(df['total transactions'],df['total product'])
    #df['average product'] = avg_prod

    #min_prod,max_prod = min_max_product(big_df,keys)
    #df['minimum product'] = min_prod
    #df['maximum product'] = max_prod
    #max_prod = max_product(big_df,keys)
    #df['maximum product'] = max_prod

    #df = df[[c for c in df if c not in ['stayed']]+['stayed']]

    #df = df.rename({'keys':'Customer Account','mean':'time mean','max':'time max','last':'time last','total':'total transactions','min':'time min','stayed':'left'},axis=1)

    #df['maximum amount'] = max_amount(big_df,keys)
    if 1 == 0:
        trans = ''
        trans_price = []
        price = 0

        big_df.pop('Date')
        big_df.pop('UPC Number')
        big_df.pop('Product Name')
        big_df.pop('Qty')
        big_df.pop('Regular Price')
        big_df.pop('Discount')

        for index,row in big_df.iterrows():
            if index % 1000 == 0:
                print(index)
            if index > 10000000:
                break
            if row['Customer Account'] == '0044885061174':
                if trans != row['Transaction Number']:
                    trans = row['Transaction Number']
                    if price != 0:
                        trans_price.append(price)
                    price = row['Sale Price']
                else:
                    price += row['Sale Price']
        
        print(trans_price)
    #df['minimum product'] = min_product(big_df,keys)
    #df['minimum amount'] = min_amount(big_df,keys)

    #df['total amount'] = total_amount(big_df,keys)
    #df['average amount'] = average_product(df['total transactions'],df['total amount'])

    #df['total discount'] = total_discount(big_df,keys)

    #df['maximum discount'] = max_discount(big_df,keys)
    #df['minimum discount'] = min_discount(big_df,keys)
    #df['average discount'] = average_product(df['total transactions'],df['total discount'])

    #df['percent discount'] = discount_div_amount(df['total discount'],df['total amount'])

    print(df)
    #print(df['total transactions'])
    df.to_pickle(fm+'_churn.pkl')
    df.to_csv(fm+'_churn.csv',index=False)
    if 1 == 0:
        for index,row in big_df.iterrows():
            if row['Discount'] < 0:
                print(big_df.iloc[index])
                break

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
    dp = []
    for i in df.keys():
        if i != 'Customer Account':
            dp.append(i)
    for i in dp:
        df.pop(i)
    for index,row in df.iterrows():
        if row['Customer Account'] not in keys:
            continue
        ind = keys.index(row['Customer Account'])
        arr[ind] += 1
    return arr

def total_amount(df,keys):
    # this function will go through the filename.pkl dataframe and count the number of products for each customer account
    # this will not count the qty, but it will only count the total separate items
    # so 10 Breads will count the same as 1 Bread
    arr = np.zeros(len(keys))
    dp = []
    for i in df.keys():
        if i != 'Customer Account' and i != 'Sale Price':
            dp.append(i)
    for i in dp:
        df.pop(i)
    for index,row in df.iterrows():
        if row['Customer Account'] not in keys:
            continue
        ind = keys.index(row['Customer Account'])
        arr[ind] += row['Sale Price']
    return arr

def total_discount(df,keys):
    # this function will go through the filename.pkl dataframe and count the number of products for each customer account
    # this will not count the qty, but it will only count the total separate items
    # so 10 Breads will count the same as 1 Bread
    arr = np.zeros(len(keys))
    dp = []
    for i in df.keys():
        if i != 'Customer Account' and i != 'Discount':
            dp.append(i)
    for i in dp:
        df.pop(i)
    for index,row in df.iterrows():
        if index % 10000 == 0:
            print(index)
        if row['Customer Account'] not in keys:
            continue
        ind = keys.index(row['Customer Account'])
        arr[ind] += row['Discount']
    return arr

def average_product(total_trans,total_prod):
    # this function will take previously generated arrays of total transactions and total purchases to 
    # calculate the average number of products purchased at each transaction
    arr = []
    for i in range(len(total_prod)):
        arr.append(total_prod[i]/total_trans[i])
    return arr

def discount_div_amount(total_discount,total_amount):
    # this function will divide every item in discount by the amount to see what percentage of purchases 
    # uses a discount
    arr = []
    for i in range(len(total_discount)):
        arr.append(total_discount[i]/total_amount[i])
    return arr

def max_product(df,keys):
    # this function will find the min and max number of items bought in one transaction
    #dic = {}
    #for i in keys:
    #    dic[i] = np.zeros(4,dtype=np.uint64)
    dic = np.zeros((len(keys),4),dtype=np.uint64)

    for index,row in df.iterrows():
        if index%10000 == 0:
            print(index)
        TN = row['Transaction Number']
        CA = row['Customer Account']
        if CA not in keys:
            continue
        CA = keys.index(CA)
        TN = np.uint64(int(TN))
        if dic[CA][1] != TN:
            dic[CA][1] = TN
            dic[CA][3] = 1
        else:
            dic[CA][3] += 1
        
        if dic[CA][3] >= dic[CA][2]:
            dic[CA][0] = dic[CA][1]
            dic[CA][2] = dic[CA][3]

    arr = []
    for i in range(len(keys)):
        arr.append(dic[i][2])
    
    return arr

def min_product(df,keys):
    # this function will find the min and max number of items bought in one transaction
    #dic = {}
    #for i in keys:
    #    dic[i] = np.zeros(4,dtype=np.uint64)
    dic = np.zeros((len(keys),4),dtype=np.uint64)
    dic = dic + 100000

    for index,row in df.iterrows():
        if index%10000 == 0:
            print(index)
        TN = row['Transaction Number']
        CA = row['Customer Account']
        if CA not in keys:
            continue
        CA = keys.index(CA)
        TN = np.uint64(int(TN))
        if dic[CA][1] != TN:
            if dic[CA][3] <= dic[CA][2]:
                dic[CA][0] = dic[CA][1]
                dic[CA][2] = dic[CA][3]
            dic[CA][1] = TN
            dic[CA][3] = 1
        else:
            dic[CA][3] += 1

    arr = []
    for i in range(len(keys)):
        arr.append(dic[i][2])
    
    return arr

def max_amount(df,keys):
    # this function will find the min and max number of items bought in one transaction
    dic = np.zeros((len(keys),4),dtype=np.float)

    for index,row in df.iterrows():
        if index%10000 == 0:
            print(index)
        TN = row['Transaction Number']
        CA = row['Customer Account']
        if CA not in keys:
            continue
        CA = keys.index(CA)
        TN = np.float(float(TN))
        SP = float(row['Sale Price'])
        #print(SP)
        if dic[CA][1] != TN:
            dic[CA][1] = TN
            dic[CA][3] = SP
        else:
            dic[CA][3] += SP
        
        if dic[CA][3] >= dic[CA][2]:
            dic[CA][0] = dic[CA][1]
            dic[CA][2] = dic[CA][3]

    arr = []
    for i in range(len(keys)):
        arr.append(dic[i][2])
    
    return arr

def min_amount(df,keys):
    # this function will find the min and max number of items bought in one transaction
    #dic = {}
    #for i in keys:
    #    dic[i] = np.zeros(4,dtype=np.uint64)
    dic = np.zeros((len(keys),4),dtype=np.float)
    dic = dic + 10000

    for index,row in df.iterrows():
        if index%10000 == 0:
            print(index)
        TN = row['Transaction Number']
        CA = row['Customer Account']
        if CA not in keys:
            continue
        CA = keys.index(CA)
        TN = np.float(float(TN))
        SP = float(row['Sale Price'])
        #print(SP)
        if dic[CA][1] != TN:
            if dic[CA][3] <= dic[CA][2]:
                dic[CA][0] = dic[CA][1]
                dic[CA][2] = dic[CA][3]
            dic[CA][1] = TN
            dic[CA][3] = SP
        else:
            dic[CA][3] += SP
        

    arr = []
    for i in range(len(keys)):
        arr.append(dic[i][2])
    
    return arr

def max_discount(df,keys):
    # this function will find the min and max number of items bought in one transaction
    dic = np.zeros((len(keys),4),dtype=np.float)
    dp = []
    for i in df.keys():
        if i != 'Customer Account' and i != 'Discount' and i != 'Transaction Number':
            dp.append(i)
    for i in dp:
        df.pop(i)
    for index,row in df.iterrows():
        if index%10000 == 0:
            print(index)
        TN = row['Transaction Number']
        CA = row['Customer Account']
        if CA not in keys:
            continue
        CA = keys.index(CA)
        TN = np.float(float(TN))
        DC = float(row['Discount'])
        #print(SP)
        if dic[CA][1] != TN:
            dic[CA][1] = TN
            dic[CA][3] = DC
        else:
            dic[CA][3] += DC
        
        if dic[CA][3] >= dic[CA][2]:
            dic[CA][0] = dic[CA][1]
            dic[CA][2] = dic[CA][3]

    arr = []
    for i in range(len(keys)):
        arr.append(dic[i][2])
    
    return arr

def min_discount(df,keys):
    # this function will find the min and max number of items bought in one transaction
    #dic = {}
    #for i in keys:
    #    dic[i] = np.zeros(4,dtype=np.uint64)
    dic = np.zeros((len(keys),4),dtype=np.float)
    dic = dic + 10000
    dp = []
    for i in df.keys():
        if i != 'Customer Account' and i != 'Discount' and i != 'Transaction Number':
            dp.append(i)
    for i in dp:
        df.pop(i)
    for index,row in df.iterrows():
        if index%10000 == 0:
            print(index)
        TN = row['Transaction Number']
        CA = row['Customer Account']
        if CA not in keys:
            continue
        CA = keys.index(CA)
        TN = np.float(float(TN))
        DC = float(row['Discount'])
        #print(SP)
        if dic[CA][1] != TN:
            if dic[CA][3] <= dic[CA][2]:
                dic[CA][0] = dic[CA][1]
                dic[CA][2] = dic[CA][3]
            dic[CA][1] = TN
            dic[CA][3] = DC
        else:
            dic[CA][3] += DC
    
    arr = []
    for i in range(len(keys)):
        arr.append(dic[i][2])
    
    return arr

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

def display_csv():
    #df = pd.read_csv('AMOHData_churn.csv')
    df = pd.read_pickle('AMOHData_churn.pkl')
    #df = pd.read_excel('AMOHData_churn.xlsx')
    #df = df.rename({'keys':'Customer Account','mean':'time mean','max':'time max','last':'time last','total':'total transactions','min':'time min'},axis=1)
    #df = df.rename({'stayed':'left'},axis=1)
    #df.pop('cdf')
    #df.to_csv('AMOHData_churn_with_nan.csv',index=False)
    #drop = []
    #for index,row in df.iterrows():
    #    if row['left'] != 0 and row['left'] != 1:
    #        drop.append(index)
    #df = df.drop(drop)
    df = df.drop(['Customer Account','time last','cdf'],axis=1)
    #df.to_excel('AMOHData_churn.xlsx',index=False)
    df.to_csv('AMOHData_churn.csv',index=False)
    print(df)

def clean_big_dataframe():
    if (len(sys.argv) > 1):
            fm = sys.argv[1]
    else:
        fm = 'AMOHData'    

    df = pd.read_pickle(fm+'_churn.pkl')
    big_df = pd.read_pickle(fm+'.pkl')

    print(big_df)
    keys = list(df['Customer Account'])
    ind_arr = np.zeros(12000000,dtype=np.uint8)
    print(ind_arr.shape)
    ind = 0
    print("0044885051831" not in keys)

    for index,row in big_df.iterrows():
        if index % 10000 == 0:
            print(index)
        if row['Customer Account'] not in keys:
            ind_arr[ind] = index
            ind += 1
    print(f'{ind = }')
    pop_arr = np.zeros(ind,np.uint8)
    for i in range(ind):
        pop_arr[i] = ind_arr[i]
    big_df = big_df.drop(pop_arr)
    print(pop_arr)
    print(big_df)
    print(f'{ind = }')
    #big_df.to_pickle(fm+'.pkl')



if __name__ == "__main__":
    start = datetime.now()

    #main()
    #display_csv()
    clean_big_dataframe()

    end = datetime.now()-start
    print(f"{end.seconds} seconds")
