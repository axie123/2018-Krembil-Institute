# This code is created and modified by Andy Xie and Tomas Tokar at Krembil in 2017.

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# The final standardized dataset and the validated training interactions are loaded.
d1 = pd.DataFrame(pd.read_csv('C:\Users'+'\\'+'final_db_standardized.txt', sep = '\t', low_memory = False))
d2 = pd.DataFrame(pd.read_csv('C:\Users'+'\\'+'training_validated_interactions.txt', sep = '\t', low_memory = False))

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

# Identifying the dependent and independent variables. 
X = ml_file.iloc[:,3:]
y = ml_file.Validated

# The multi-class labels are converted into appropriate binary labels. 
binr = preprocessing.LabelBinarizer() 
binr.fit(y)
y = binr.transform(y) 

# Spliting the training and testing data.
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Logistical Regression
lr = linear_model.LogisticRegression() 

# Fitting the training set.
lr.fit(X_train,y_train)

# Prodiciting the probability of the test set having it's current state.
prb = lr.predict_proba(X_test) 

# Putting this into a datatable.
pt = pd.DataFrame({'TF_Target' : X_test.index, 'proba' : prb[:,1], 'truth' : y_test[:,0]})
pt = pt.sort_values(by = 'proba', ascending = False) #//

# F1 Score Analysis
idx = np.linspace(1, np.log10(pt.shape[0]), 20) #//
f1_score = list()
for i in idx:
    spt = pt.head(np.int(10**i))
    # False negatives
    fn = np.sum(~d2['Factor / Gene Name'].isin(spt.TF_Target))   # False negative rate
    # Precision and Recall
    prc = np.mean(spt.truth == 1)
    rec = np.sum(spt.truth == 1) * 1. / fn
    # F1 score
    f1 = (prc * rec * 1.) / (prc + rec)
    # Add to list
    f1_score.append(f1)

# Pack togethe rinto data frame

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

