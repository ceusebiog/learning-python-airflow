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

def analyze_transactions():
  """Función para analizar transacciones y detectar patrones sospechosos."""
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

if __name__ == '__main__':
    analyze_transactions()