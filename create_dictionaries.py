import pandas as pd
import pickle
from datetime import datetime

if __name__ == '__main__':

    start = datetime.now()

    filename = 'SD.pkl'

    timestamp_to_datetime = lambda x : (datetime.strptime(x,'%Y-%m-%d %H:%M:%S.%f'))
    

    df = pd.read_pickle(filename)

    all_time_dict ={}
    time_diff = {}
    last_time_shopping = {}

    ta = ""

    for index, row in df.iterrows():
        CA = df['Customer Account'][index]
        if (ta == row['Transaction Number']):
            continue
        else:
            ta = row['Transaction Number']
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



    #file_one = open(filename[0:-4]+
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
