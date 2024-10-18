import pandas as pd
from sqlalchemy import create_engine
import os

DB_USER = os.getenv('POSTGRES_USER', 'airflow')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'airflow')
DB_NAME = os.getenv('POSTGRES_DB', 'airflow')
DB_HOST = os.getenv('POSTGRES_HOST', 'postgres')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
print(DATABASE_URL)
engine = create_engine(DATABASE_URL)

def extract_data(file_path):
    """Función para extraer datos del archivo CSV."""
    return pd.read_csv(file_path)

def transform_data(df):
    """Función para transformar los datos."""
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

def load_data(df, table_name):
    """Función para cargar los datos en PostgreSQL."""
    df.to_sql(table_name, con=engine, if_exists='append', index=False)
    print(f'Datos cargados en la tabla {table_name}')

if __name__ == '__main__':
    # Ruta al archivo CSV
    file_path = '/app/transactions.csv'
    
    # ETL
    # Extract
    data = extract_data(file_path)
    # Transform
    transformed_data = transform_data(data)
    # Load
    load_data(transformed_data, 'transactions')