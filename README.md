# NJIT-REU

All of the pickle files that are needed should be in the Binary Files folder in the shared Google Drive. 
While it is possible to generate the same data by using the python files and the data files that were originally 
share with us, it took my computer quite a while to create those files with those python programs, like 10 mins 
for the 'Sale Data.csv' and about 30 mins for the 'AMOHData.csv'. 

In order to create those pickle files, I first used loading_supermarket_data.py to create the first pkl file with 
no suffixes, so 'AMOHData.pkl' and 'Sale Data.pkl'. I created the different .pkl dictionary files with create_dictionaries.py
and then I use load_dicts.py to load those files into different programs using the definition load_dicts. I used clean.py 
to create the .pkl file that'll hold the dictionary of the times in between visits for each customer by converting the 
deltatimes into floats, 1 being a day and (1/(24*60*60)) being a second. I then used those arrays with floats to check 
the type of distribution for each customer by using the kstest and checking the p-value returned from that function. 
The distribution will be used to calculate if the customer has stayed or left already and that'll become the target 
attribute. 

In these programs I have been using different packages like pandas, numpy, matplotlib, pickle, scipy, datetime, and sys. 
I downloaded these packages using a program called pip, but there are other ways of downloading these packages too. 

In most of the programs I uploaded, I start the program with "start = datetime.now()" and then I end the code with 
printing the difference between the start of the program and the end to determine how long the program takes to run. 
I also have most of the code in the if statement "if __name__ == '__main__':" because this code will only be run 
when that specific file is ran. This is so that if I import the file into another file, such as how I import load_dicts, 
then the main code ran in the program is not executed in the file that its imported to. For example, any file that imports
load_dicts.py will ignore any of the code under the "if __name__ == '__main__':" statement and it'll only load the functions.




