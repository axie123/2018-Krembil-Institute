import numpy as np
import pandas as pd
import os as os 

# Merging Software
def expand_1(data):
    x = data['Sample ID'].str.split(',')
    x = x.values.tolist()
    l = [len(v) for v in x]
    final = pd.DataFrame({'Sample ID': np.concatenate(x)}, data.index.repeat(l))
    final = final.join(data.drop('Sample ID', 1)).reset_index(drop=True)
    return(final)

def expand_2(data):
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
    expanded_1 = expand_1(data1)
    expanded_2 = expand_2(data2)
    x = merge(expanded_1,expanded_2)
    return(x)

# Importing and Cleaning
def importandparse_first(fila): 
    files = open(fila,"r")
    counter = 0
    lines = files.readlines()
    for row_num in range(len(lines)):
        if 'Sample ID' not in lines[row_num]:
            counter += 1
        else:
            skipper = counter
            dataset = pd.read_csv(fila,sep = '\t', skiprows = skipper)
            dataset_final = pd.DataFrame(dataset)
            return(dataset_final)

def importandparse_second(fila): 
    files = open(fila,"r")
    counter = 0
    lines = files.readlines()
    for row_num in range(len(lines)):
        if 'Samples' not in lines[row_num]:
            counter += 1
        else:
            skipper = counter
            dataset = pd.read_csv(fila,sep = '\t', skiprows = skipper)
            dataset_final = pd.DataFrame(dataset)
            return(dataset_final)

# Row adding
def rowmerge(data1,data2):
    frames = [data1] + [data2]
    final = pd.concat(frames,ignore_index = True)
    return(final)



# Final Code
directory = 'C:\Users\Databases\JurisicaLab'

counter = 0
for filename in os.listdir(directory):
    sets = (directory + "\\" + filename)
    for set_files in os.listdir(sets):
        file_loc = (sets + "\\" + set_files)
        list_sets = []
        for i in os.listdir(file_loc):
            if i.endswith(".txt"):
                list_sets += [os.path.join(file_loc, i)]
                if len(list_sets) == 2:
                    counter += 1
                    data1 = importandparse_first(list_sets[0])
                    data2 = importandparse_second(list_sets[1])
                    combine = combined(data1,data2)
                    number = 'combined' + str(counter)
                    combine.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final' + "\\" + number + '.txt', sep = '\t')

typed = {'Sample ID':str,'Factor':str,'Cell line/Tissue':str,'Treatment':str,'Main Accession':str,
         'Sub Accession':str,'Software':str,'Reference':str,'Pubmed':str,'Gene ID':str,'Gene Name':str,
         'Type in ChIPBase':str,'Gene Type':str,'Factor Number':str,'Factors':str,'Sample Number':str}

base1 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final' + "\\" + 'final1.txt', sep = '\t', low_memory = False))
base2 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final' + "\\" + 'final2.txt', sep = '\t', low_memory = False))
initial_merge = rowmerge(base1,base2)
lista = [initial_merge]
for i in os.listdir('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final')[2:]:
    print(i)
    lista += [pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final' + "\\" + i, sep = '\t', low_memory = False))]
    if len(lista) == 2:
        data1 = lista[0]
        data2 = lista[1]
        merged = rowmerge(data1,data2)
        lista = [merged]
merged.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final' + "\\" + 'final_db.txt', sep = '\t')

