from airflow.operators.python_operator import PythonOperator

from airflow.models import DAG
from datetime import datetime, timedelta
import os
import sys
sys.path.append('../../work/Record_Union/')
import fraud_detection

fd = fraud_detection.fraud_detection()

default_args = {
    'owner': 'airflow',
    'start_date':  datetime.today() + timedelta(minutes=5),
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}
dag = DAG(
    dag_id='fraud_detection', default_args=default_args)


load_stream.set_upstream(streams)

load_stream = PythonOperator(
    task_id='load_stream',
    provide_context=True,
    python_callable=fd.load_stream,
    params ={'path': os.path.join('/disk/ru', 'streams')},
    dag=dag)

label = PythonOperator(
    task_id='label',
    provide_context=True,
    python_callable=fd.label,
    dag=dag)
label.set_upstream(load_stream)

load_users = PythonOperator(
    task_id='load_users',
    provide_context=True,
    python_callable=fd.load_users,
    params ={'path':os.path.join('/disk/ru', 'users')},
    dag=dag)

build_model = PythonOperator(
    task_id='build_model',
    provide_context=True,
    python_callable=fd.build_model,
    dag=dag)

build_model.set_upstream([load_stream,load_users,label])
