import pandas as pd
import numpy as np

dt = {'Patient': ['A', 'A', 'B', 'A', 'B'],
      'Area': [200,200,180,200,180],
      'Year': [2010,2011,2012,2013,2014],
      'College': [0.064, 0.121, 0.145, 0.180, 0.231]}

dtf = pd.DataFrame(dt)
jut = dtf.pivot(index = 'Patient', columns = 'Year')
ju = pd.DataFrame(jut.values)
ju['Patient'] = list(jut.index.values)
ju.set_index('Patient',inplace = True)
print(ju)

final = dtf.pivot(index = 'Patient', columns = 'Year', values = 'Area')
final2 = dtf.pivot(index = 'Patient', columns = 'Year', values = 'College')
g = pd.DataFrame(final.values)
g2 = pd.DataFrame(final2.values)
g['Patient'] = list(final.index.values)
g.set_index('Patient',inplace = True)
g2['Patient'] = list(final.index.values)
g2.set_index('Patient',inplace = True)
print(g)
print(g2)

'''
final.columns = final.columns.droplevel()
final.reset_index(drop = True)

lista = list(final.columns.values)

new_df = pd.DataFrame(final[('Area', 2010L)].values)
new_df.columns = ['2010 Area']

j = 1
for i in lista[1:]:
    new_df['201' + str(j) +' Area'] = pd.DataFrame(final[i].values)
    j += 1
    if j == 5:
         break
k = 0
for a in lista[5:]:
    new_df['201' + str(k) +' College'] = pd.DataFrame(final[a].values)
    k += 1
    if k == 5:
        break 



lista = list(final.columns.values)
j = 0
for i in lista:
    final['201' + str(j) +' Area'] = final[i]
    del(final[i])
    j += 1
    if j == 5:
         break
k = 0
for a in lista[5:]:
    final['201' + str(k) +' College'] = final[a]
    del(final[a])
    k += 1
    if k == 5:
        break
'''



