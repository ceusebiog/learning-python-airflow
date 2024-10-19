import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

"""Función para extraer datos del archivo CSV."""
def extract_data(file_path):
  return pd.read_csv(file_path)

"""Función para transformar los datos."""
def transform_data(df):
  df['transaction_date'] = pd.to_datetime(df['transaction_date'])
  return df

"""Función para preparar el conjunto de datos para entrenamiento."""
def prepare_dataset(file_path):
  data = extract_data(file_path)

  df = transform_data(data)

  df['is_fraud'] = 0
  df.loc[df['amount'] < -5000, 'is_fraud'] = 1
  df.loc[df['amount'] > 20000, 'is_fraud'] = 1
  
  df['transaction_type_encoded'] = df['transaction_type'].astype('category').cat.codes
  df['hour'] = pd.to_datetime(df['transaction_date']).dt.hour
  df['is_night'] = (df['hour'] >= 22) | (df['hour'] <= 6)

  return df[['amount', 'transaction_type_encoded', 'is_night', 'is_fraud']]

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

"""Función para realizar el ajuste fino del modelo."""
def fine_tune_model(df):
  X = df[['amount', 'transaction_type_encoded', 'is_night']]
  y = df['is_fraud']
  
  # Definir el modelo base
  model = RandomForestClassifier(random_state=42)
  
  # Definir los hiperparámetros para la búsqueda aleatoria
  param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
  }
  
  # Configurar RandomizedSearchCV
  random_search = RandomizedSearchCV(
    model, 
    param_distributions=param_dist,
    n_iter=100,
    scoring='f1',
    cv=5,
    random_state=42,
    n_jobs=-1
  )
  
  # Ajustar el modelo
  random_search.fit(X, y)
  
  print(f"Mejores parámetros: {random_search.best_params_}")
  return random_search.best_estimator_

"""Función para evaluar el modelo en el conjunto de prueba."""
def test_model_evaluation(test_df, model):
  # Seleccionar las características del dataset de prueba
  X_test = test_df[['amount', 'transaction_type_encoded', 'is_night']]
  y_test = test_df['is_fraud']
  
  # Hacer predicciones con el modelo entrenado
  y_pred = model.predict(X_test)
  
  # Imprimir el reporte de clasificación para evaluar el rendimiento
  print("Evaluación del modelo en el dataset independiente:")
  print(classification_report(y_test, y_pred))

"""Función para implementar un modelo de stacking."""
def stacking_model(df):
  X = df[['amount', 'transaction_type_encoded', 'is_night']]
  y = df['is_fraud']
  
  # Dividir en entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Definir los modelos base
  estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
  ]
  
  # Definir el modelo de stacking con Logistic Regression como meta-modelo
  stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
  )
  
  # Entrenar el modelo
  stacking_clf.fit(X_train, y_train)
  
  # Evaluar el modelo
  print("Evaluación en conjunto de prueba:")
  test_model_evaluation(X_test, y_test, stacking_clf)
  
  return stacking_clf

"""Función para implementar un Voting Classifier."""
def voting_model(df):
  X = df[['amount', 'transaction_type_encoded', 'is_night']]
  y = df['is_fraud']
  
  # Dividir en entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Definir los modelos base
  clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
  clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
  clf3 = LogisticRegression(max_iter=1000)
  
  # Crear el Voting Classifier
  voting_clf = VotingClassifier(
    estimators=[('rf', clf1), ('gb', clf2), ('lr', clf3)],
    voting='soft'  # Usa 'soft' para promediar las probabilidades
  )
  
  # Entrenar el modelo
  voting_clf.fit(X_train, y_train)
  
  # Evaluar el modelo
  print("Evaluación en conjunto de prueba:")
  test_model_evaluation(X_test, y_test, voting_clf)
  
  return voting_clf

if __name__ == '__main__':
  file_path_train = '/app/transactions_train.csv'
  file_path_test = '/app/transactions_test.csv'

  df = prepare_dataset(file_path_train)

  # validate_model(df)

  # model = train_model(df)

  # tune_model = fine_tune_model(df)

  # stacking_clf = stacking_model(df)

  # Entrenar y evaluar el modelo de voting
  voting_clf = voting_model(df)

  test_df = prepare_dataset(file_path_test)

  # test_model(test_df, tune_model)

  test_model_evaluation(test_df, voting_clf)