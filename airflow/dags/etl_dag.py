# airflow/dags/etl_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
import os

# Configuración de la conexión a PostgreSQL
DB_USER = os.getenv('POSTGRES_USER', 'user')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
DB_HOST = os.getenv('POSTGRES_HOST', 'postgres')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')
DB_NAME = os.getenv('POSTGRES_DB', 'bank_db')

DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(DATABASE_URL)

def extract_data(file_path):
    return pd.read_csv(file_path)

def transform_data(df):
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    return df

def load_data(df, table_name):
    df.to_sql(table_name, con=engine, if_exists='append', index=False)

def etl_process():
    file_path = '/usr/local/airflow/dags/transactions.csv'  # Cambiar la ruta si es necesario
    data = extract_data(file_path)
    transformed_data = transform_data(data)
    load_data(transformed_data, 'transactions')

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG('bank_fraud_etl', default_args=default_args, schedule_interval='@daily', catchup=False) as dag:
    etl_task = PythonOperator(
        task_id='run_etl_process',
        python_callable=etl_process,
    )

    etl_task