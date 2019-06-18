import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

mdl = Sequential()

mdl.add(Dense(200,activation = 'relu',input_shape = ()))
mdl.add(Dense(200,activation = 'relu',input_shape = ()))
mdl.add(Dense(200,activation = 'relu',input_shape = ()))
mdl.add(Dense(1))
