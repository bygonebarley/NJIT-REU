import pandas as pd
things = open("System User id.txt").readline().split(',')

clean_df = pd.read_pickle('Sale Data.pkl')

print(clean_df['Date'])
print(clean_df['Discount'])

