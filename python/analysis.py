import pandas as pd
from sqlalchemy import create_engine
import os

DB_USER = os.getenv('POSTGRES_USER', 'user')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
DB_HOST = os.getenv('POSTGRES_HOST', 'postgres')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'bank_db')

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(DATABASE_URL)

"""Función para analizar transacciones y detectar patrones sospechosos."""
def analyze_transactions():
  query = "SELECT * FROM transactions;"
  df = pd.read_sql(query, con=engine)
  
  # Análisis básico
  print("Resumen de transacciones:")
  print(df.describe())

  # Análisis de transacciones inusuales
  suspicious_transactions = df[(df['amount'] < -1000) | (df['amount'] > 10000)]
  if not suspicious_transactions.empty:
    print("Transacciones sospechosas:")
    print(suspicious_transactions)

"""Función para preparar el conjunto de datos para entrenamiento."""
def prepare_dataset():
    query = "SELECT * FROM transactions;"
    df = pd.read_sql(query, con=engine)
    
    df['is_fraud'] = 0
    df.loc[df['amount'] < -5000, 'is_fraud'] = 1
    df.loc[df['amount'] > 20000, 'is_fraud'] = 1
    
    return df

if __name__ == '__main__':
    df = prepare_dataset()
    print(df.sort_values(by=['is_fraud'], ascending=False).head())