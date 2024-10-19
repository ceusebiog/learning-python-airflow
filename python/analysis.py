import pandas as pd
from sqlalchemy import create_engine
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
  
  # Afinar el modelo con GridSearchCV
  model = tune_model(X_train, y_train)
  
  # Hacer predicciones
  y_pred = model.predict(X_test)
  print(classification_report(y_test, y_pred))
    
  return model

def tune_model(X_train, y_train):
  """Función para afinar el modelo con GridSearchCV."""
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

if __name__ == '__main__':
    df = prepare_dataset()
    model = train_model(df)