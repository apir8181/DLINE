
NAME=$1
DATADIR=./raw_data/${NAME}/data

mkdir ./data/${NAME}
python ./python_script/format_asu_edges.py ${DATADIR} ./data/${NAME}

shuf ./data/${NAME}/edges.txt -o ./data/${NAME}/edges.txt_tmp
mv ./data/${NAME}/edges.txt_tmp ./data/${NAME}/edges.txt
