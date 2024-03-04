# clases y funciones del pipeline de preprocesamiento.
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from keras.models import load_model



data = pd.read_csv("dfmlops.csv")
from sklearn.model_selection import train_test_split
X = data[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
       'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
       'Humidity3pm', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'RainToday','WindGustDir', 'WindDir9am', 'WindDir3pm']]

y = data['RainfallTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X,y.values.reshape(-1,1),test_size=0.2,random_state = 42)

def feat_eng(X_train, X_test, y_train, y_test):
    '''Rellenar valores faltantes luego del split para evitar fuga de datos'''
    # Inicializar instancias
    numeric_imputer = SimpleImputer(strategy='mean') 
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Separar numericas de categóricas
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    #  Rellenar faltantes sin fuga de datos
    X_train[numeric_features] = numeric_imputer.fit_transform(X_train[numeric_features])
    X_train[categorical_features] = categorical_imputer.fit_transform(X_train[categorical_features])
    X_test[numeric_features] = numeric_imputer.fit_transform(X_test[numeric_features])
    X_test[categorical_features] = categorical_imputer.fit_transform(X_test[categorical_features])

    return X_train, X_test, y_train, y_test



    
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    '''
    Estandarización z-score
    '''
    def __init__(self):
        self.scaler = StandardScaler()
    def fit(self, X_train, y=None):
        
        numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns

        # Inicilizar
        self.scaler = StandardScaler()

        # Fittear
        self.scaler.fit(X_train[numeric_features])

        return self

    def transform(self, X):
        # Identificar numericas
        numeric_features = X.select_dtypes(include=['float64', 'int64']).columns

        # Transform only numeric features
        X_numeric = X[numeric_features]
        X_transformed = self.scaler.transform(X_numeric)
        X_numeric_scaled = pd.DataFrame(X_transformed, columns=X_numeric.columns)

        # Concatenar numericas con categoricas
        X_result = pd.concat([X_numeric_scaled, X.select_dtypes(exclude='number')], axis=1)

        return X_result
    
class CustomDummiesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoding = None
        
    def fit(self, X_train, y_train=None):
        self.encoding = pd.get_dummies(X_train)
        return self

    def transform(self, X):
        # Conseguir dummies mediante one-hot encoding
        X_encoded = pd.get_dummies(X)
        # Alinear las columnas
        X_encoded = X_encoded.align(self.encoding, axis=1, fill_value=0)[0]
        
        # Convertir las columnas booleanas a enteros (0 o 1)
        boolean_columns = X_encoded.select_dtypes(include=bool).columns
        X_encoded[boolean_columns] = X_encoded[boolean_columns].astype(int)
        
        return X_encoded
    

class Load_Keras_Model():
    def __init__(self, filepath):
        self.load_model(filepath)
        self.filepath = filepath
        
    def fit(self, X=None, y=None):
        self.load_model(self.filepath)
        return self

    def predict(self, X):
        input_df = pd.DataFrame(X)
        input_df.to_csv('debug_df.csv', index=False)
        if self.model is None:
            raise ValueError('Model not loaded')
        predictions = self.model.predict(X)
        return predictions

    def load_model(self, filepath):
        '''
        Carga el modelo desde un archivo dado por el filepath.
        '''
        self.model = load_model(filepath)

# Creamos un pipeline para cada tipo de modelo
pipe_regression = Pipeline([
    ('zscore_scaler', CustomStandardScaler()), 
    ('dummies_encoder', CustomDummiesEncoder()),
    ('neural_network', Load_Keras_Model(filepath='reg_nn_model.h5'))
])
pipe_classification = Pipeline([
    ('zscore_scaler', CustomStandardScaler()),
    ('dummies_encoder', CustomDummiesEncoder()),
    ('neural_network', Load_Keras_Model(filepath='best_model_optuna_cla.h5'))
])

X_train, X_test, y_train, y_test = feat_eng(X_train, X_test, y_train, y_test)
pipe_regression.fit(X_train, y_train)
y_train_bin = np.where(y_train != 0, 1, 0)
pipe_classification.fit(X_train, y_train_bin)
