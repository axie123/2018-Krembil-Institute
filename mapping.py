import numpy as np
import pandas as pd

# Creates the dictionary and dataframe required for the recent data.
file1 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\cell lines(3).txt',sep = '\t',header = None,na_values = '-'))
raw_list = list(file1[0])
correct_list = list(file1[3])
dictn = {}
for i in range(len(raw_list)):
    dictn[raw_list[i]] = correct_list[i]
    
# Cleaning the used data.
def expand(data):
    x = data['Cell line/Tissue'].str.split(',')
    x = x.values.tolist()
    l = [len(v) for v in x]
    final = pd.DataFrame({'Cell line/Tissue': np.concatenate(x)}, data.index.repeat(l))
    final = final.join(data.drop('Cell line/Tissue', 1)).reset_index(drop=True)
    return(final)

big_file = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1\Final' + '\\' + 'final_db_cleaned.txt',sep = '\t',low_memory=False))
del(big_file['Unnamed: 0'])
final = big_file.dropna(subset = ['Cell line/Tissue'])
final['Cell line/Tissue'] = final['Cell line/Tissue'].map(dictn)
final = final.dropna(subset = ['Cell line/Tissue'])

final.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1\Final' + "\\" + 'final_db_cleaned_final.txt', sep = '\t')

'''
file1 = file1.assign(A = file1[0], B = file1[1], C = file1[2], D = file1[3], E = file1[4])

data = pd.DataFrame(raw_list)
data['Entry.correct'] = correct_list

big_file = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1' + '\\' + 'testingfile.txt',sep = '\t',low_memory=False))
b = expand(big_file)
print(b)

big_file = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1\Final' + '\\' + 'final_db.txt',sep = '\t',low_memory=False))
f = big_file['Software']
f.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1' + "\\" + 'testingfile.txt', sep = '\t')

d = {'a':[1,[2,5],3,4],
     'b':[1,2,3,4],
     'c':[1,2,3,4]}

df = pd.DataFrame(d)
'''
