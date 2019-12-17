import pandas as pd
import numpy as np

def rowmerge(data1,data2):
    frames = []
    dataset1 = pd.DataFrame(data1)
    dataset2 = pd.DataFrame(data2)
    frames = [dataset1] + [dataset2]
    final = pd.concat(frames)
    return(final)

typed = {'Sample ID':str,'Factor':str,'Cell line/Tissue':str,'Treatment':str,'Main Accession':str,
         'Sub Accession':str,'Software':str,'Reference':str,'Pubmed':str,'Gene ID':str,'Gene Name':str,
         'Type in ChIPBase':str,'Gene Type':str,'Factor Number':str,'Factors':str,'Sample Number':str}

base1 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final5' + "\\" + 'combined81.txt', sep = '\t', low_memory = False))
base2 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final5' + "\\" + 'combined82.txt', sep = '\t', low_memory = False))
initial_merge = rowmerge(base1,base2)
lista = [initial_merge]
for i in os.listdir('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final5')[2:]:
    print(i)
    lista += [pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final5' + "\\" + i, sep = '\t', low_memory = False))]
    if len(lista) == 2:
        data1 = lista[0]
        data2 = lista[1]
        merged = rowmerge(data1,data2)
        lista = [merged]
merged.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final5' + "\\" + 'final5.txt', sep = '\t')

# Elementary row operators. Backup only.
def addrows(newrow,data):
    dataset = pd.DataFrame(data)
    cols = ''
    for i in dataset.columns.values:
        cols += i
    new_drs = pd.DataFrame([newrow],columns = list(cols))
    print(new_drs)
    new_dataset = dataset.append(new_drs,ignore_index = True)
    return(new_dataset)

def getrows(data):
    dataset = pd.DataFrame(data)
    holder = []
    newrows = []
    for i in dataset.columns.values:
        holder += [i]
    length = len(holder[0])
    for j in range(length):
        newrows = [dataset[j:j+1]]
    return(newrows)

lista = []
for i in os.listdir('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final')[2:0]:
    print(i)
    lista += [i]
    if len(lista) == 2:
        data1 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final' + "\\" + lista[0], sep = '\t', low_memory = False))
        data2 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final' + "\\" + lista[1], sep = '\t', low_memory = False))
        merged = rowmerge(data1,data2)
        merged.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Final' + "\\" + 'final.txt', sep = '\t')
        lista = ['final.txt']
 
def getdict(data):
    new_dict = {}
    dataset = pd.DataFrame(data)
    for i in dataset.columns.values:
        new_dict[i] = data1[i]
    return(new_dict)

def getrows(new_dict):
    newrows = []
    for i in new_dict:
        newrows += [new_dict[i]]
    return(newrows)
