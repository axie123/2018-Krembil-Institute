import numpy as np
import pandas as pd

def importandparse(fila): 
    files = open(fila,"r")
    counter = 0
    lines = files.readlines()
    for row_num in range(len(lines)):
        if 'Sample ID' not in lines[row_num]:
            counter += 1
            print(counter)
        else:
            skipper = counter
            dataset = pd.read_csv(fila,sep = '\t', skiprows = skipper)
            dataset_final = pd.DataFrame(dataset)
            return(dataset_final)

print(importandparse("C:\Users\Databases\JurisicaLab\Series I\Set 1\ChIPBase_v2.0-information_of_bulkySample_2018_05_25_pm_32_53.txt"))


'''
fila = "C:\Users\Databases\JurisicaLab\Series I\Set 3\ChIPBase_v2.0-information_of_bulkySample_2018_05_25_pm_37_55.txt"
files = open(fila,"r")
lines = files.readlines()
counter = 0
for row_num in range(len(lines)):
    if 'Sample ID' not in lines[row_num]:
        counter += 1
    print(counter)
print(len(lines))

for i, line in enumerate(files):
    print(lines)
'''

'''
def importandparse(files): 
    dataset = pd.read_csv(files,sep = '\t', skiprows = 0)
    dataset_final = pd.DataFrame(dataset)
    counter = 0
    for row_num in range(len(dataset_final)):
        lista = list(dataset_final.iloc[row_num])
        if 'Sample ID' not in lista:
            counter += 1
        else:
            skipper = counter + 1
            dataset = pd.read_table(files,sep = '\t', skiprows = skipper)
            dataset_final = pd.DataFrame(dataset)
            return(dataset_final)
'''        
