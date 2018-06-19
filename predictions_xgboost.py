#! /usr/bin/env python
# -*- coding: utf-8 -*-

import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

x = pd.read_csv('x_values.csv', header=None).values
y = pd.read_csv('y_values.csv', header=None)[0].values
model = XGBRegressor(depth=3,
                     min_child_weight=1,
                     subsample=1,
                     colsample_bytree=1,
                     objective='reg:linear',
                     n_estimators=100,
                     learning_rate=0.1)

kfold = KFold(n_splits=10, random_state=0)
y_pred = cross_val_predict(model, x, y, cv=kfold)
print np.corrcoef(y, y_pred)
