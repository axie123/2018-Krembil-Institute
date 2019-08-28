# This code is created and modified by Andy Xie and Tomas Tokar at Krembil in 2017.

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loads the main training dataset.
fname = 'C:\Users' + "\\" + 'final_db_standardized.txt'
d = pd.read_csv(fname, sep = '\t')

# Loads all the validated training interactions. 
fname = 'C:\Users' + "\\" + 'training_validated_interactions.txt'
v = pd.read_csv(fname, sep = '\t')

# Trimming and Splicing the dataset for machine learning.
d = d[d['Main Accession'] != 'ENCODE']
d['TF_Target'] = d[['Factor', 'Gene Name']].apply(lambda x: '_'.join(x), axis=1)
v['TF_Target'] = v[['TF', 'Target']].apply(lambda x: '_'.join(x), axis=1)

# Extracting the TF-DNA pairs.
ds = d[['TF_Target', 'Main Accession']]
ds = ds.drop_duplicates()
ds['Detected'] = 1

# Reshaping the dataframe.
ds_wide = ds.pivot(index = 'TF_Target', columns = 'Main Accession', values = 'Detected')
ds_wide = ds_wide.fillna(value = 0)

# Getting the validated TF-Target interactions.
tf_target = ds_wide.index.to_series()
tf_target = tf_target.str.split("_", expand = True)
tf_target.columns = ['TF', 'Target']
tf_target['Validated'] = tf_target.index.isin(v['TF_Target'])
tf_target['Validatable'] = tf_target['TF'].isin(v.TF) & tf_target['Target'].isin(v.Target)
tf_target[['Validatable', 'Validated']].apply(pd.Series.value_counts)

ds_wide = pd.concat([tf_target, ds_wide], axis = 1)
ml_ready = ds_wide[ds_wide.Validatable]

# Splitting the dependent and independent variables. 
x = ml_ready.iloc[:,4:]
y = ml_ready.Validated

# Turning the multi-class labels into binary labels. 
lb = preprocessing.LabelBinarizer()
lb.fit(y)
y = lb.transform(y)

# Splitting the training and testing dataset.
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.6,random_state = 1)

# Logistic Regression and Predicting the probability of the state.
lm = linear_model.LogisticRegression()
lm.fit(x_train, y_train)
p = lm.predict_proba(x_test)

# Putting the predictions into a data table.
pt = pd.DataFrame({'TF_Target' : x_test.index, 'proba' : p[:,1], 'truth' : y_test[:,0]})
pt = pt.sort_values(by = 'proba', ascending = False)
idx = np.linspace(1, np.log10(pt.shape[0]), 20)

f1_score = list()
for i in idx:
    spt = pt.head(np.int(10**i))

    # False negative rate
    fn = np.sum(~v.TF_Target.isin(spt.TF_Target))

    # Precision and Recall
    prc = np.mean(spt.truth == 1)
    rec = np.sum(spt.truth == 1) * 1. / fn

    # F1 score
    f1 = (prc * rec * 1.) / (prc + rec)

    # Add to list
    f1_score.append(f1)

# Pack together into dataframe.
res = pd.DataFrame({'N': idx, 'F_score' : f1_score})

def calc_fscore(b):

    # Predicted positives and negatives
    ppos = x_test.index[b == 1.]
    pneg = x_test.index[b == 0.]

    # True positives
    tp = np.sum(ppos.isin(v.TF_Target))

    # False positives
    fp = np.sum(pneg.isin(v.TF_Target))

    # False negatives
    fn = np.sum(~v.TF_Target.isin(ppos))

    # Precision and Recall
    prc = tp * 1. / (tp + fp)
    rec = tp * 1. / (tp + fn)

    # F1 score
    f1 = (prc * rec * 1.) / (prc + rec)

    return f1

res_univariate = pd.DataFrame({'N' : np.log10(x_test.sum(axis = 0)), 'F_score' : x_test.apply(calc_fscore)})
res_univariate = res_univariate.dropna()

plt.plot(res_univariate.N, res_univariate.F_score, 'go')
plt.plot(res.N, res.F_score, 'bo')
plt.plot(res.N, res.F_score, 'r--')
plt.xlabel('N [log10]')
plt.ylabel('F1 score')
plt.show()
