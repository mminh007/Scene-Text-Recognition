from airflow import DAG
from airflow.operators.bash import BashOperator
import datetime as dt


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
    
    data_processing_task = BashOperator(task_id="data_processing",
                                   bash_command="python C://Users/tuanm/OneDrive/project2/Scene_Text_Recognition/processing.py")
    
    train_yolo_task = BashOperator(task_id="yolo_model",
                                   bash_command=("python C:/Users/tuanm/OneDrive/project2/Scene_Text_Recognition/yolo_train.py"))
    
    train_crnn_task = BashOperator(task_id="text_recognition_model",
                                    bash_command=("python C://Users/tuanm/OneDrive/project2/Scene_Text_Recognition/train.py" \
                                                    "--config-file=C://Users/tuanm/OneDrive/project2/Scene_Text_Recognition/configs/base_parameters.yaml" ))
    
    
data_processing_task >> train_yolo_task >> train_crnn_task