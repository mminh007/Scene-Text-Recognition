from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import datetime as dt
from pathlib import Path


def create_temp_dir():
    """
    Create artifacts_dir on cloud
    """
    path_save = Path("./str_tmp_dir") 
    path_save.mkdir(parents=True, exist_ok=True)
    if not path_save.exists():
        raise FileNotFoundError(f"Failed to create directory: {path_save}")
    

args = {
    "owner": "admin",
    "start_date": dt.datetime(2022, 12, 1),
    "retries": 1,
    "retry_delays": dt.timedelta(minutes=1),
    "depends_on_past": False
}


with DAG(
    dag_id="Scene_Text_Recognition",
    default_args=args,
    schedule_interval=None,
    tags=["yolo", "crnn"],
) as dag:
    
    create_temp_dir_task = PythonOperator(task_id="create_temp_dir",
                                          python_callable=create_temp_dir,
                                          dag=dag)
    
    data_processing_task = BashOperator(task_id="data_processing",
                                   bash_command=("python C://Users/tuanm/OneDrive/project2/Scene_Text_Recognition/processing.py" \
                                                 "--config-file=C://Users/tuanm/OneDrive/project2/Scene_Text_Recognition/configs/base_parameters.yaml"))
    
    train_yolo_task = BashOperator(task_id="yolo_model",
                                   bash_command=("python C:/Users/tuanm/OneDrive/project2/Scene_Text_Recognition/yolo_train.py" \
                                                 "--config-file=C://Users/tuanm/OneDrive/project2/Scene_Text_Recognition/configs/base_parameters.yaml"))
    
    train_crnn_task = BashOperator(task_id="text_recognition_model",
                                    bash_command=("python C://Users/tuanm/OneDrive/project2/Scene_Text_Recognition/train.py" \
                                                    "--config-file=C://Users/tuanm/OneDrive/project2/Scene_Text_Recognition/configs/base_parameters.yaml" ))
    
    register_model_task  = BashOperator(task_id="register_model",
                                        bash_command=("python C://Users/tuanm/OneDrive/project2/Scene_Text_Recognition/register.py" \
                                                      "--config-file=C://Users/tuanm/OneDrive/project2/Scene_Text_Recognition/configs/base_parameters.yaml"))


create_temp_dir >> data_processing_task >> train_yolo_task >> train_crnn_task >> register_model_task