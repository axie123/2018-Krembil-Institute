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

# Below are the elementary functions that align the charts and the merging by rows. These will be for back up only.    
'''
def merge(name1,name2):
    dataset = [pd.ExcelFile(name1),pd.ExcelFile(name2)]
    convdict = [dict(dataset[0]),dict(dataset[1])]
    tables = [pd.DataFrame(convdict[0]),pd.DataFrame(convdict[1])]
    table0names = []
    table1names = []
    for colname in (tables[0].columns.values):
        table0names += [colname]
    for colname in (tables[1].columns.values):
        table1names += [colname]
    unknown = []
    for i in table1names:
        if i in table0names:
            table0names.remove(i)
        else:
            unknown += [i]
    final_names = table0names + table1names
    for j in unknown:
        (tables[0])[j] = (tables[1])[j]
    return(tables[0])

def alignset(name1,name2):
    count = 0
    counter = 0
    for noms in name1['Sample ID']:
        count += 1
    for noms in name2['Samples']:
        counter += 1
    if count < counter:
        return("Data 1 was short.")
    elif count > counter:
        return("Data 2 was short.")

def alignment(name1,name2):
    count = 0
    counter = 0
    for noms in name1['Sample ID']:
        count += 1
    for noms in name2['Samples']:
        counter += 1
    col_count = 0
    fill = []
    if count < counter:
        for i in name1.columns.values:
            col_count += 1
        for j in range(col_count):
            fill += ['NaN']
        for k in range(counter - count):
            name1.loc[count + k] = fill
        return(name1)
    elif count > counter:
        for i in name2.columns.values:
            col_count += 1
        for j in range(col_count):
            fill += ['NaN']
        for k in range(count - counter):
            name2.loc[counter + k] = fill
        return(name2)
    else:
        return(name1)
'''



            
    
    


