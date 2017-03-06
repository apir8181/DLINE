DATA_DIR=/home/qiaoan/distributed_LINE/test_data

echo "generate data"
python ./python_script/most_popular_flickr.py ./data/flickr/group.txt ./data/flickr/group_popular_5.txt
echo "10%"
python ./python_script/eval_multilabel.py ${DATA_DIR}/vector.txt ./data/flickr/group_popular_5.txt 0.1
echo "30%"
python ./python_script/eval_multilabel.py ${DATA_DIR}/vector.txt ./data/flickr/group_popular_5.txt 0.3
echo "50%"
python ./python_script/eval_multilabel.py ${DATA_DIR}/vector.txt ./data/flickr/group_popular_5.txt 0.5
