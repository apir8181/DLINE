echo "10%"
python ./python_script/eval_multilabel.py ./data/blog_catalog/vector.txt ./data/blog_catalog/group.txt 0.1
echo "30%"
python ./python_script/eval_multilabel.py ./data/blog_catalog/vector.txt ./data/blog_catalog/group.txt 0.3
echo "50%"
python ./python_script/eval_multilabel.py ./data/blog_catalog/vector.txt ./data/blog_catalog/group.txt 0.5
