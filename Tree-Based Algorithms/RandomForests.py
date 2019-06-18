import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
'''
fname = 'C:\Users\Andy Xie\Documents\Work\Research\Databases\MI' + "\\" + 'final_db_standardized.txt'
d = pd.read_csv(fname, sep = '\t')

fname = 'C:\Users\Andy Xie\Documents\Work\Research\Databases\MI' + "\\" + 'training_validated_interactions.txt'
v = pd.read_csv(fname, sep = '\t')

d = d[d['Main Accession'] != 'ENCODE']
d['TF_Target'] = d[['Factor', 'Gene Name']].apply(lambda x: '_'.join(x), axis=1)
v['TF_Target'] = v[['TF', 'Target']].apply(lambda x: '_'.join(x), axis=1)
ds = d[['TF_Target', 'Main Accession']]
ds = ds.drop_duplicates()
ds['Detected'] = 1

ds_wide = ds.pivot(index = 'TF_Target', columns = 'Main Accession', values = 'Detected')
ds_wide = ds_wide.fillna(value = 0)

tf_target = ds_wide.index.to_series()
tf_target = tf_target.str.split("_", expand = True)
tf_target.columns = ['TF', 'Target']
tf_target['Validated'] = tf_target.index.isin(v['TF_Target'])
tf_target['Validatable'] = tf_target['TF'].isin(v.TF) & tf_target['Target'].isin(v.Target)
tf_target[['Validatable', 'Validated']].apply(pd.Series.value_counts)

ds_wide = pd.concat([tf_target, ds_wide], axis = 1)
ml_file = ds_wide[ds_wide.Validatable]

X = ml_file.iloc[:,4:]
y = ml_file.Validated
binr = preprocessing.LabelBinarizer()
binr.fit(y)
y = binr.transform(y)
'''
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.9,random_state = 1)

tree = RandomForestRegressor(n_estimators = 50, max_depth = 200, max_features = 200, min_samples_leaf = 0.0000001, random_state = 1) # lr
m = tree.fit(X_train,y_train)
pred = cross_val_predict(m,X_test,y_test,cv = 15) # prb
pt = pd.DataFrame({'TF_Target' : X_test.index, 'proba' : pred, 'truth' : y_test[:,0]})
pt = pt.sort_values(by = 'proba', ascending = False) #//

lm = linear_model.LogisticRegression()
lm.fit(X_train, y_train)
p = lm.predict_proba(X_test)
ptt = pd.DataFrame({'TF_Target' : X_test.index, 'proba' : p[:,1], 'truth' : y_test[:,0]})
ptt = ptt.sort_values(by = 'proba', ascending = False)

# F1 Score Analysis
idx = np.linspace(1, np.log10(pt.shape[0]), 20) #//
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

idxx = np.linspace(1, np.log10(ptt.shape[0]), 20)
f1_score2 = list()
for i in idxx:
    sptt = ptt.head(np.int(10**i))
    # False negative rate
    fnn = np.sum(~v.TF_Target.isin(sptt.TF_Target))
    # Precision and Recall
    prc = np.mean(sptt.truth == 1)
    rec = np.sum(sptt.truth == 1) * 1. / fnn
    # F1 score
    f1 = (prc * rec * 1.) / (prc + rec)
    # Add to list
    f1_score2.append(f1)

# Pack togethe rinto data frame

res = pd.DataFrame({'N': idx, 'F_score' : f1_score})
res2 = pd.DataFrame({'N': idxx, 'F_score2' : f1_score2})
def calc_fscore(b):
    # Predicted positives and negatives
    ppos = X_test.index[b == 1.] #//
    pneg = X_test.index[b == 0.]
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

def calc_fscore2(b):
    # Predicted positives and negatives
    ppos = X_test.index[b == 1.] #//
    pneg = X_test.index[b == 0.]
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

res_univariate = pd.DataFrame({'N' : np.log10(X_test.sum(axis = 0)), 'F_score' : X_test.apply(calc_fscore)})
res_univariate = res_univariate.dropna()
res_univariate2 = pd.DataFrame({'N' : np.log10(X_test.sum(axis = 0)), 'F_score' : X_test.apply(calc_fscore2)})
res_univariate2 = res_univariate2.dropna()

# Plot results
plt.plot(res_univariate.N, res_univariate.F_score, 'go')
plt.plot(res_univariate2.N, res_univariate2.F_score, 'ko')
plt.plot(res.N, res.F_score, 'bo')
plt.plot(res.N, res.F_score, 'r--')
plt.plot(res2.N, res2.F_score2, 'bo')
plt.plot(res2.N, res2.F_score2, 'y--')
plt.xlabel('N [log10]')
plt.ylabel('F1 score')
plt.show()
