@echo off
call conda activate tensorflow
echo "========= 環境啟動成功 ========="

python run_experiments.py

echo "全部執行完畢！"
pause