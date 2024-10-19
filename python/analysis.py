import pandas as pd
import numpy as np
import os

from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE


DB_USER = os.getenv('POSTGRES_USER', 'user')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
DB_HOST = os.getenv('POSTGRES_HOST', 'postgres')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'bank_db')

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(DATABASE_URL)

"""Función para preparar el conjunto de datos para entrenamiento."""
def prepare_dataset():
  query = "SELECT * FROM transactions;"
  df = pd.read_sql(query, con=engine)
  
  df['is_fraud'] = 0
  df.loc[df['amount'] < -5000, 'is_fraud'] = 1
  df.loc[df['amount'] > 20000, 'is_fraud'] = 1
  
  df['transaction_type_encoded'] = df['transaction_type'].astype('category').cat.codes
  df['hour'] = pd.to_datetime(df['transaction_date']).dt.hour
  df['is_night'] = (df['hour'] >= 22) | (df['hour'] <= 6)

  return df[['amount', 'transaction_type_encoded', 'is_night', 'is_fraud']]

"""Función para entrenar un modelo de clasificación de fraude."""
def train_model(df):
  # Seleccionar las características y la variable objetivo
  X = df[['amount', 'transaction_type_encoded', 'is_night']]
  y = df['is_fraud']
  
  # Dividir el dataset en entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
  # Aplicar SMOTE para balancear las clases
  smote = SMOTE(random_state=42)
  X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

  # Afinar el modelo con GridSearchCV
  model = tune_model(X_train_balanced, y_train_balanced)
  
  # Hacer predicciones
  y_pred = model.predict(X_test)
  print(classification_report(y_test, y_pred))
    
  return model

"""Función para afinar el modelo con GridSearchCV."""
def tune_model(X_train, y_train):
  param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
  }
  rf = RandomForestClassifier(random_state=42)
  grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
  grid_search.fit(X_train, y_train)
  print("Mejores parámetros:", grid_search.best_params_)
  return grid_search.best_estimator_

"""Función para validar el modelo usando validación cruzada."""
def validate_model(df):
  X = df[['amount', 'transaction_type_encoded', 'is_night']]
  y = df['is_fraud']
  
  # Aplicar SMOTE para balancear las clases
  smote = SMOTE(random_state=42)
  X_balanced, y_balanced = smote.fit_resample(X, y)

  # Crear el modelo usando los mejores hiperparámetros encontrados
  model = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=42)

  # Validación cruzada con 5 particiones
  cv_scores = cross_val_score(model, X_balanced, y_balanced, cv=5, scoring='accuracy')

  print("Puntuaciones de validación cruzada:", cv_scores)
  print("Precisión promedio:", np.mean(cv_scores))
  
  return cv_scores

if __name__ == '__main__':
    df = prepare_dataset()

    validate_model(df)

    model = train_model(df)