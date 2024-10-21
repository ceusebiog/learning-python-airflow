import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sqlalchemy import create_engine


DB_USER = os.getenv('POSTGRES_USER', 'user')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
DB_HOST = os.getenv('POSTGRES_HOST', 'postgres')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'bank_db')

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

def main():
  df = prepare_dataset()

  print(df.head())

  voting_model(df)

def prepare_dataset():
  engine = create_engine(DATABASE_URL)
  query = "select * from transactions where transaction_date >= current_date - interval '1 day' and transaction_date < current_date;"
  df = pd.read_sql(query, con=engine)

  df['is_fraud'] = 0
  df.loc[df['amount'] < -5000, 'is_fraud'] = 1
  df.loc[df['amount'] > 20000, 'is_fraud'] = 1
  
  df['transaction_type_encoded'] = df['transaction_type'].astype('category').cat.codes
  df['hour'] = pd.to_datetime(df['transaction_date']).dt.hour
  df['is_night'] = (df['hour'] >= 22) | (df['hour'] <= 6)

  return df[['amount', 'transaction_type_encoded', 'is_night', 'is_fraud']]

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

def test_model_evaluation(test_df, model):
  # Seleccionar las características del dataset de prueba
  X_test = test_df[['amount', 'transaction_type_encoded', 'is_night']]
  y_test = test_df['is_fraud']
  
  # Hacer predicciones con el modelo entrenado
  y_pred = model.predict(X_test)
  
  # Imprimir el reporte de clasificación para evaluar el rendimiento
  print("Evaluación del modelo en el dataset independiente:")
  print(classification_report(y_test, y_pred))