echo "1%"
python ./python_script/eval_multilabel.py ./vector.txt ./data/youtube/group.txt 0.01
echo "5%"
python ./python_script/eval_multilabel.py ./vector.txt ./data/youtube/group.txt 0.05
echo "10%"
python ./python_script/eval_multilabel.py ./vector.txt ./data/youtube/group.txt 0.1
