# This code is created and modified by Andy Xie and Tomas Tokar at Krembil in 2018.

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

d1 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\MI'+'\\'+'final_db_standardized.txt', sep = '\t', low_memory = False))
d2 = pd.DataFrame(pd.read_csv('C:\Users\Andy Xie\Documents\Work\Research\Databases\MI'+'\\'+'training_validated_interactions.txt', sep = '\t', low_memory = False))

d1 = d1[d1['Main Accession'] != 'ENCODE']
d1['Factor / Gene Name'] = d1['Factor'] + ' / ' + d1['Gene Name']
d2['Factor / Gene Name'] = d2['TF'] + ' / ' + d2['Target']
taken = d1[['Factor / Gene Name','Main Accession']]
taken = taken.drop_duplicates()
taken['Verified'] = np.ones(len(taken['Factor / Gene Name']), dtype = np.int)
taken_wide_f = taken.pivot(index = 'Factor / Gene Name', columns = 'Main Accession',values = 'Verified')
taken_wide_f = taken_wide_f.fillna(0)
target_table = (taken_wide_f.index.to_series()).str.split(" / ",expand = True)
target_table.columns = ['Factor', 'Gene Name']
target_table['Validated'] = target_table.index.isin(d2['Factor / Gene Name'])
target_table['Validatable'] = (target_table['Factor'].isin(d2.TF)) & (target_table['Gene Name'].isin(d2.Target))
final_wide = pd.concat([target_table,taken_wide_f], axis = 1)
ml_file = final_wide[final_wide.Validatable == 1]

X = ml_file.iloc[:,4:]
y = ml_file.Validated
binr = preprocessing.LabelBinarizer()
binr.fit(y)
y = binr.transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 1)

bag = GradientBoostingRegressor(n_estimators = 200,max_depth = 200, random_state = 1)
m = bag.fit(X_train,y_train)
pred = cross_val_predict(m,X_test,y_test,cv = 10) # prb
pt = pd.DataFrame({'TF_Target' : X_test.index, 'proba' : pred, 'truth' : y_test[:,0]})
pt = pt.sort_values(by = 'proba', ascending = False) #//lm = linear_model.LogisticRegression()

# F1 Score Analysis
idx = np.linspace(1, np.log10(pt.shape[0]), 20) #//
f1_score = list()
for i in idx:
    spt = pt.head(np.int(10**i))
    # False negative rate
    fn = np.sum(~d2['Factor / Gene Name'].isin(spt.TF_Target))
    # Precision and Recall
    prc = np.mean(spt.truth == 1)
    rec = np.sum(spt.truth == 1) * 1. / fn
    # F1 score
    f1 = (prc * rec * 1.) / (prc + rec)
    # Add to list
    f1_score.append(f1)

# Pack together into data frame

res = pd.DataFrame({'N': idx, 'F_score' : f1_score})
def calc_fscore(b):
    # Predicted positives and negatives
    ppos = X_test.index[b == 1.] #//
    pneg = X_test.index[b == 0.]
    # True positives
    tp = np.sum(ppos.isin(d2['Factor / Gene Name']))
    # False positives
    fp = np.sum(pneg.isin(d2['Factor / Gene Name']))
    # False negatives
    fn = np.sum(~d2['Factor / Gene Name'].isin(ppos))
    # Precision and Recall
    prc = tp * 1. / (tp + fp)
    rec = tp * 1. / (tp + fn)
    # F1 score
    f1 = (prc * rec * 1.) / (prc + rec)
    return f1

res_univariate = pd.DataFrame({'N' : np.log10(X_test.sum(axis = 0)), 'F_score' : X_test.apply(calc_fscore)})
res_univariate = res_univariate.dropna()

# Plot results
plt.plot(res_univariate.N, res_univariate.F_score, 'go')
plt.plot(res.N, res.F_score, 'bo')
plt.plot(res.N, res.F_score, 'r--')
plt.xlabel('N [log10]')
plt.ylabel('F1 score')
plt.show()
