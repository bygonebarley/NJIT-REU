import pandas as pd
import pickle
from datetime import datetime
import sys

if __name__ == '__main__':

    start = datetime.now()

    if (len(sys.argv) > 1):
        filename = sys.argv[1]
    else:
        filename = 'AM.pkl'

    timestamp_to_datetime = lambda x : (datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f'))
    

    df = pd.read_pickle(filename)

    all_time_dict ={}
    time_diff = {}
    last_time_shopping = {}
    transaction_numbers = {}

    for index, row in df.iterrows():
        CA = df['Customer Account'][index]
        TN = row['Transaction Number']
    
        if TN in transaction_numbers:
            transaction_numbers[TN][1].append(row['Date'])
            if (transaction_numbers[TN][1][-1]-transaction_numbers[TN][1][0]).seconds > 60:
                print(f"Longer than a minute - Transaction # {TN}")
            all_time_dict[transaction_numbers[TN][0]][-1] = transaction_numbers[TN][1][int(len(transaction_numbers[TN][1])/2)]
            continue
        else:
            transaction_numbers[TN] = [CA,[row['Date']]]

        '''
        if CA in transaction_numbers:
            if TN in transaction_numbers[CA]:
                continue
            else:
                transaction_numbers[CA].append(TN)
        else:
            transaction_numbers[CA] = [TN]
        '''
        #if (len(df['Date'][index]) < 10):
        #    continue
        if CA in all_time_dict:
            all_time_dict[CA].append(df['Date'][index])
            #all_time_dict[CA].append(timestamp_to_datetime(df['Date'][index]))
        else:
            all_time_dict[CA] = [df['Date'][index]]
            #all_time_dict[CA] = [timestamp_to_datetime(df['Date'][index])]

    for values in all_time_dict:
        time_diff[values] = []
        for index in range(len(all_time_dict[values])-1):
            time_diff[values].append(all_time_dict[values][index+1]-all_time_dict[values][index])
        last_time_shopping[values] = df.iloc[-1]['Date'] - all_time_dict[values][-1]
        #last_time_shopping[values] = timestamp_to_datetime(df.iloc[-1]['Date']) - all_time_dict[values][-1]



    file_one = open(filename[0:-4]+'_all_time.pkl','wb')
    file_two = open(filename[0:-4]+'_time_diff.pkl','wb')
    file_three = open(filename[0:-4]+'_last_time.pkl','wb')

    pickle.dump(all_time_dict,file_one)
    pickle.dump(time_diff,file_two)
    pickle.dump(last_time_shopping,file_three)

    file_one.close()
    file_two.close()
    file_three.close()

    code_time = (datetime.now()-start)
    print(str(code_time.seconds) + " seconds")
