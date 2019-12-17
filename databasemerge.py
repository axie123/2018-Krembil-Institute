import pandas as pd
import numpy as np

data1 = {'Sample ID': ['a','b','c','d','e','f'],
         'val1': [5,6,7,8,9,10],
         'val2': [9,10,11,12,13,14]}

data2 = {'val3': [14,15,16,17,18],
         'val4': [19,20,21,22,23],
         'Samples': ['a','b','c','x','y']}         

base1 = pd.DataFrame(data1)
base2 = pd.DataFrame(data2)

def expand(data):
    x = data['Samples'].str.split(',')
    x = x.values.tolist()
    l = [len(v) for v in x]
    final = pd.DataFrame({'Samples': np.concatenate(x)}, data.index.repeat(l))
    final = final.join(data.drop('Samples', 1)).reset_index(drop=True)
    return(final)

def merge(data1,data2):
    vals = {}
    for i in data2:
        if i == 'Samples':
            vals['Sample ID'] = data2[i]
            del(data2[i])
            data2['Sample ID'] = vals['Sample ID']
    final_t = pd.merge(data1,data2,how = 'outer',on = 'Sample ID')
    return(final_t)

def combined(data1,data2):
    base1 = pd.DataFrame(data1)
    base2 = pd.DataFrame(data2)
    expanded_2 = expand(base2)
    x = merge(base1,expanded_2)
    return(x)

print(combined(base1,base2))
