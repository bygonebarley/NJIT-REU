import pandas as pd
from datetime import datetime
import numpy as np
import math
import sys

if __name__ == '__main__':
    # this is just to calculate how long the file takes to load
    start = datetime.now()

    # I tried using this to change it from timestamps to datetime but it didn't work
    timestamp_to_datetime = lambda x : (datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S.%f'))

    # This was for if the file contained a header or not, if it did then the header would be row 0 and if not it would be None
    head = 0 # 0 or None

    if (len(sys.argv) > 1):
        path = sys.argv[1]
        filename = path[0:-4]
    else:
        # this is the file name that would be loaded and used to save the database as pickle data
        filename = 'test'
        path = filename + '.csv'
    # this loads the csv file with the proper variable types for each of the columns
    # this also provides a header if one is not given
    df = pd.read_csv(path,dtype = {'Data':str,'Transaction Number':str,'UPC Number':str,'Product Name':str,'Customer Account':str,'Qty':float,'Regular Price':float,'Sale Price':float,'Discount':float},header=head,names=['Date','Transaction Number','UPC Number','Product Name','Customer Account','Qty','Regular Price','Sale Price','Discount']) #, parse_dates = ['Date'],infer_datetime_format = False)


    #customer_codes = {}
    # this would load in the store accounts into an array 
    ignore_codes = open('System User id.txt').readline().split(',')

    drop = []
    # this forloop was used to obtain all of the rows that should be cleaned out because the customer account was unknown or a store account
    replace = []

    system_user = 0
    not_ignored = 0

    for i in range(len(df)):

        if df['Customer Account'][i] in ignore_codes:
            if df['Customer Account'][i] not in replace:
                replace.append(df['Customer Account'][i])
            system_user += 1
            continue
            #df['Customer Account'][i] = 'System User ID'

        if type(df['Customer Account'][i]) == type(1.0):
            if df['Customer Account'][i] not in replace:
                replace.append(df['Customer Account'][i])
            system_user += 1
            continue
            #df['Customer Account'][i] = 'Null'

        if type(df['Date'][i]) == type(1.0):
            drop.append(i)
            system_user += 1
            continue
        if (len(df['Date'][i]) < 10):
            drop.append(i)
            system_user += 1
            continue
        not_ignored += 1
        #if df['Customer Account'][i] in customer_codes:
        #    customer_codes[df['Customer Account'][i]].append(df['Date'][i])
        #else:
        #    customer_codes[df['Customer Account'][i]] = [df['Date'][i]]

    # this line gets rid of all of the unwanted rows
    cleaned_df = df.drop(df.index[drop])

    cleaned_df['Customer Account'] = cleaned_df['Customer Account'].replace(replace,'misc')

    cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])

    # this saves the dataframe to a binary file
    cleaned_df.to_pickle(filename+'.pkl')
    #cleaned_df.to_csv('bleh_amoh.csv')

    print(f"There are {system_user} ignored codes")
    print(f"There are {not_ignored} not ignored codes")
    #print(len(customer_codes))

    # this prints out the time it took to load the file
    code_time = (datetime.now()-start)
    print(str(code_time.seconds) + " seconds")
