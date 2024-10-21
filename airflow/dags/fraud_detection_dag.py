import pandas as pd

from airflow.decorators import dag, task
from airflow.models import Variable

from datetime import datetime, timedelta

from sqlalchemy import create_engine

default_args = {
  'owner': 'airflow',
  'depends_on_past': False,
  'start_date': datetime(2024, 10, 18),
  'email_on_failure': False,
  'email_on_retry': False,
  'retries': 1,
  'retry_delay': timedelta(minutes=5),
}

@dag(
  dag_id='fraud_detection_pipeline',
  description='Pipeline para cargar datos y entrenar el modelo de detección de fraude',
  default_args=default_args,
)
def taskflow():
  @task()
  def load_data():
    DB_USER = Variable.get('POSTGRES_USER', 'user')
    DB_PASSWORD = Variable.get('POSTGRES_PASSWORD', 'password')
    DB_HOST = Variable.get('POSTGRES_HOST', 'postgres')
    DB_PORT = Variable.get('POSTGRES_PORT', '5432')
    DB_NAME = Variable.get('POSTGRES_DB', 'bank_db')
    DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

    # Conectar a PostgreSQL
    engine = create_engine(DATABASE_URL)
    
    # Supongamos que tenemos un archivo CSV con las transacciones
    year = datetime.now().year
    month = datetime.now().month
    day = datetime.now().day - 1

    df = pd.read_csv(f'/opt/airflow/dags/transaction_{day}_{month}_{year}.csv')
    
    # Cargar el DataFrame en PostgreSQL
    df.to_sql('transactions', engine, if_exists='append', index=False)
    print("Datos cargados correctamente en PostgreSQL")
  
  def train_model():
    import fraud_analysis
    fraud_analysis.main()
  
  load_data()

taskflow()