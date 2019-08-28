# This code is created and modified by Andy Xie and Tomas Tokar at Krembil in 2018.

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Imports the main processed file and the validated interactions file. Trims them as well.
ml_file = pd.DataFrame(pd.read_csv('C:\Users\ml_ready.txt', sep = '\t', low_memory = False))
d2 = pd.DataFrame(pd.read_csv('C:\Users'+'\\'+'training_validated_interactions.txt', sep = '\t', low_memory = False))
del(ml_file['Unnamed: 0'])
d2['Factor / Gene Name'] = d2['TF'] + ' / ' + d2['Target']
ml_file = ml_file.set_index('Factor / Gene Name')
ml_file.drop(['Factor', 'Gene Name'],axis = 1,inplace = True)

# Dividing the dependent and independent variables.
X = ml_file.iloc[:,2:]
y = ml_file.Verified

# Turning multi-label data into binary data.
binr = preprocessing.LabelBinarizer() 
binr.fit(y)
y = binr.transform(y) 

# Spliting the training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2)

# Setting up the parameters of the ridge regression.
ridge = Ridge(alpha = 0.000001, normalize = True)
ridge.fit(X_train,y_train)
ridge_pred = ridge.predict(X_test)

# Placing my predicted data into a table.
pt = pd.DataFrame({'TF_Target' : X_test.index, 'ridge_pred' : ridge_pred[:,0], 'truth' : y_test[:,0]})
pt = pt.sort_values(by = 'ridge_pred', ascending = False) #//

idx = np.linspace(1, np.log10(pt.shape[0]), 20) 
f1_score = list()
for i in idx:
    spt = pt.head(np.int(10**i))
    # Using the d2 data to predict false negative rate.
    fn = np.sum(~d2['Factor / Gene Name'].isin(spt.TF_Target))   # False negative rate
    # Precision and Recall
    prc = np.mean(spt.truth == 1)
    rec = np.sum(spt.truth == 1) * 1. / fn
    # F1 score
    f1 = (prc * rec * 1.) / (prc + rec)
    # Add to list
    f1_score.append(f1)

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
