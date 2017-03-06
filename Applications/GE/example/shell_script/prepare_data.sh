NAME=$1
SPLIT_NUM=$2

echo "formating edges $NAME"
rm -r ./data/$NAME
mkdir -p ./data/$NAME
python ./python_script/format_asu_edges.py ./raw_data/$NAME/data ./data/$NAME

echo "sorting"
sort ./data/$NAME/edges.txt -n -k 1,1 -o ./data/$NAME/edges.txt_tmp
mv ./data/$NAME/edges.txt_tmp ./data/$NAME/edges.txt
#shuf ./data/$NAME/edges.txt -o ./data/$NAME/edges.txt_tmp
#mv ./data/$NAME/edges.txt_tmp ./data/$NAME/edges.txt

echo "spliting data ${SPLIT_NUM} partition"
rm ~/distributed_LINE/test_data/edges_part/*
split -n l/${SPLIT_NUM} ./data/${NAME}/edges.txt
count=0
for f in ./x*
do
    mv $f ~/distributed_LINE/test_data/edges_part/edges.txt_part_${count} 
    count=$(( $count + 1 ))
done

for HOST in "pc57" "pc61" "pc63" "pc51" "pc52" "pc53" "pc54"
do
    echo "copy to $HOST"
    scp -r ~/distributed_LINE/test_data/edges_part ${HOST}:~/distributed_LINE/test_data/
    scp ./data/${NAME}/degree.txt ${HOST}:~/distributed_LINE/test_data/
done
