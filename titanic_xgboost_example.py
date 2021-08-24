# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:09:53 2020

@author: adeel
"""

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Cargar datos
train_df = pd.read_csv(r'.\train.csv', header=0)
test_df = pd.read_csv(r'.\test.csv', header=0)

# Llenar valores faltantes con promedios o cero.
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']
nonnumeric_columns = ['Sex']

# Unión de los sets de entrenamiento y prueba.
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

# Transforma categórico en números
le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# Prepara las entradas para el modelo
train_X = big_X_imputed[0:train_df.shape[0]].to_numpy()
test_X = big_X_imputed[train_df.shape[0]::].to_numpy()
train_y = train_df['Survived']

# Profundidad de árbol, estimadores, aprendizaje.
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

print(gbm.score(train_X, train_y))
print(gbm.feature_importances_)

# Formato de salida
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv(r'.\submission.csv', index=False)
