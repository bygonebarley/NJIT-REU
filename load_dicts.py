import pandas as pd
import pickle
from datetime import datetime

def load_dicts(filename):
    # these lines create file objects for the different binary files containing the dictionaries
    file_one = open(filename+'_all_time.pkl','rb')
    file_two = open(filename+'_time_diff.pkl','rb')
    file_three = open(filename+'_last_time.pkl','rb')
    
    # this loads the files into dictionaries
    one = pickle.load(file_one)
    two = pickle.load(file_two)
    three = pickle.load(file_three)
    return one,two,three

if __name__ == '__main__':
    start = datetime.now()

    filename = 'twosupermarket_test'

    [all_time_dict,time_diff,last_time_shopping] = load_dicts(filename)

    first_key = list(all_time_dict.keys())[0]
    
    print(first_key)
    print(all_time_dict[first_key][0])
    print(time_diff[first_key][0])
    print(last_time_shopping[first_key])

    code_time = (datetime.now()-start)
    print(str(code_time.seconds) + " seconds")
