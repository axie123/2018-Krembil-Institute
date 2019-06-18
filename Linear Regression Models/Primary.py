import numpy as np
import pandas as pd

d1 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\MI'+'\\'+'final_db_standardized.txt', sep = '\t', low_memory = False))
d2 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\MI'+'\\'+'training_validated_interactions.txt', sep = '\t', low_memory = False))

# Get rid of the ENCODE in Main Accession:
d1 = d1[d1['Main Accession'] != 'ENCODE']

# Creating the merged column.
d1['Factor / Gene Name'] = d1['Factor'] + ' / ' + d1['Gene Name']
d2['Factor / Gene Name'] = d2['TF'] + ' / ' + d2['Target']

# Wide created.
taken = d1[['Factor / Gene Name','Main Accession']]
taken = taken.drop_duplicates()
taken['Verified'] = np.ones(len(taken['Factor / Gene Name']), dtype = np.int)
taken_wide_f = taken.pivot(index = 'Factor / Gene Name', columns = 'Main Accession',values = 'Verified')
taken_wide_f = taken_wide_f.fillna(0)

# Creating comparison table
target_table = (taken_wide_f.index.to_series()).str.split(" / ",expand = True)
target_table.columns = ['Factor', 'Gene Name']
target_table['Validated'] = target_table.index.isin(d2['Factor / Gene Name'])
target_table['Validatable'] = (target_table['Factor'].isin(d2.TF)) & (target_table['Gene Name'].isin(d2.Target))

# Final Wide.
final_wide = pd.concat([target_table,taken_wide_f], axis = 1)
ml_file = final_wide[final_wide.Validatable == 1]
ml_file.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\MI' + "\\" + 'ml_ready.txt', sep = '\t')
d2.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\MI' + "\\" + 'assist.txt', sep = '\t')

'''
# Manipulating Data for SVL
d1 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases' + '\\' + 'SVL.txt', sep = '\t', low_memory = False))
d2 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases' + '\\' + 'training_validated_interactions.txt', sep = '\t', low_memory = False))
del(d1['Unnamed: 0'])

d1['Factor / Gene Name'] = d1['Factor'] + ' / ' + d1['Gene Name']
del(d1['Factor'])
del(d1['Gene Name'])
columns = list(d1.columns.values[len(d1.columns.values) - 1:]) + list(d1.columns.values[0:len(d1.columns.values) - 1])
d1 = d1[columns]

d2['Factor / Gene Name'] = d2['TF'] + ' / ' + d2['Target']

factor_gene = d1["Factor / Gene Name"]
factor_gene = factor_gene.str.split(" / ", expand = True)
factor_gene.columns =  ['Factor','Gene Name']

# SVL 1: Validiation and Validiable.

factor_gene['Verified'] = d1['Factor / Gene Name'].isin(d2['Factor / Gene Name']) #//
factor_gene['Verifiable'] = factor_gene['Factor'].isin(d2['TF']) & factor_gene['Gene Name']
factor_gene = factor_gene.replace([True,False],[1,0])

merged = pd.concat([factor_gene,d1], axis = 1)
ml_ready = merged[merged.Verifiable == 1] #//
ml_ready.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Software\Multi-Factor Analysis' + "\\" + 'ml_ready.txt', sep = '\t')
'''
'''
d1 = d1[d1['Main Accession'] != 'ENCODE']
f_s.columns = ['col1'] # The merged cols for the large dataset.
s_s.columns = ['col2'] # The merged cols for the reference.
'''
