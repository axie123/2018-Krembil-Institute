import numpy as np
import pandas as pd

dtf = pd.DataFrame(pd.read_csv('C:\Users' + '\\' + 'fdcf.txt', sep = '\t', low_memory = False))
dtf['Validation'] = np.ones((len(dtf),1))
dtff = pd.DataFrame(dtf['Factor'])
dtff['Gene Name'] = dtf['Gene Name']
dtff['Cell line/Tissue'] = dtf['Cell line/Tissue']
dtff['Validation'] = dtf['Validation']
dtff['Factor / Gene Name'] = dtff['Factor'] + ' / ' + dtff['Gene Name']
del(dtff['Factor'])
del(dtff['Gene Name'])
final = dtff.drop_duplicates(subset = None, keep = 'first', inplace = False)

jutland = final.pivot(index = 'Factor / Gene Name', columns = 'Cell line/Tissue', values = 'Validation')
jutland.reset_index(level = 0, inplace = True)
jutland['Split'] = jutland['Factor / Gene Name'].str.split(' / ')
jutland['Factor'] = jutland.Split.str.get(0)
jutland['Gene Name'] = jutland.Split.str.get(1)
del(jutland['Split'])
del(jutland['Factor / Gene Name'])
columns = list(jutland.columns.values[len(jutland.columns.values) - 2:]) + list(jutland.columns.values[0:len(jutland.columns.values) - 2])
jut = jutland[columns]
final = jut.fillna(0)
final.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1\Final' + "\\" + 'MI2.txt', sep = '\t')

'''
dtf['Validation'] = np.ones((len(dtf),1))
dtff = pd.DataFrame(dtf['Factor'])
dtff['Gene Name'] = dtf['Gene Name']
dtff['Software'] = dtf['Software']
dtff['Validation'] = dtf['Validation']
dtff['Factor / Gene Name'] = dtff['Factor'] + ' / ' + dtff['Gene Name']
del(dtff['Factor'])
del(dtff['Gene Name'])
final = dtff.drop_duplicates(subset = None, keep = 'first', inplace = False)

jutland = final.pivot(index = 'Factor / Gene Name', columns = 'Software', values = 'Validation')
jutland.reset_index(level = 0, inplace = True)
jutland['Split'] = jutland['Factor / Gene Name'].str.split(' / ')
jutland['Factor'] = jutland.Split.str.get(0)
jutland['Gene Name'] = jutland.Split.str.get(1)
del(jutland['Split'])
del(jutland['Factor / Gene Name'])
columns = list(jutland.columns.values[len(jutland.columns.values) - 2:]) + list(jutland.columns.values[0:len(jutland.columns.values) - 2])
jut = jutland[columns]
final = jut.fillna(0)
final.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1\Final' + "\\" + 'MI1.txt', sep = '\t')

dtf['Validation'] = np.ones((len(dtf),1))
dtff = pd.DataFrame(dtf['Factor'])
dtff['Gene Name'] = dtf['Gene Name']
dtff['Cell line/Tissue'] = dtf['Cell line/Tissue']
dtff['Validation'] = dtf['Validation']
dtff['Factor / Gene Name'] = dtff['Factor'] + ' / ' + dtff['Gene Name']
del(dtff['Factor'])
del(dtff['Gene Name'])
final = dtff.drop_duplicates(subset = None, keep = 'first', inplace = False)

jutland = final.pivot(index = 'Factor / Gene Name', columns = 'Cell line/Tissue', values = 'Validation')
jutland.reset_index(level = 0, inplace = True)
jutland['Split'] = jutland['Factor / Gene Name'].str.split(' / ')
jutland['Factor'] = jutland.Split.str.get(0)
jutland['Gene Name'] = jutland.Split.str.get(1)
del(jutland['Split'])
del(jutland['Factor / Gene Name'])
columns = list(jutland.columns.values[len(jutland.columns.values) - 2:]) + list(jutland.columns.values[0:len(jutland.columns.values) - 2])
jut = jutland[columns]
final = jut.fillna(0)
final.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1\Final' + "\\" + 'MI2.txt', sep = '\t')

['Unnamed: 0' 'Software' 'Sample ID' 'Factor' 'Cell line/Tissue'
 'Treatment' 'Main Accession' 'Sub Accession' 'Reference' 'Pubmed'
 'Gene ID' 'Gene Name' 'Type in ChIPBase' 'Gene Type' 'Factor Number'
 'Factors' 'Sample Number' 'Validation']

jutland = final.pivot(index = 'Factor', columns = 'Software', values = 'Validation')
fg = list(final.columns.values)
jut = pd.DataFrame(jutland.values)
jut['Factor'] = list(jutland.index.values)
jut.set_index('Factor',inplace = True)
jut.columns = fg
jut1 = jut
jut1.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1\Final' + "\\" + 'MI1.txt', sep = '\t')

jutland = final.pivot(index = 'Gene Name', columns = 'Cell line/Tissue', values = 'Validation')
fg = list(final.columns.values)
jut = pd.DataFrame(jutland.values)
jut['Gene Name'] = list(jutland.index.values)
jut.set_index('Gene Name',inplace = True)
jut.columns = fg
jut1 = jut
jut1.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1\Final' + "\\" + 'MI2.txt', sep = '\t')

dtf = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1\Final' + '\\' + 'MI2.txt', sep = '\t', low_memory = False))
final = dtf.fillna(0)
final.to_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\Database 1\Final' + "\\" + 'final_MI2.txt', sep = '\t')

'''
