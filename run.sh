/home/zmp/venvs/data_acquisition/bin/activate
TASK_NAME=`cat task_name`
echo $TASK_NAME
python main.py --convlstm -task_name $TASK_NAME