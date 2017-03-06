
../../../build/Applications/GE/graphembedding \
    -graph_part_file ./data/flickr/edges.txt \
    -output_file ./data/flickr/vector.txt \
    -embedding_size 100 \
    -sample_edges_millions 330 \
    -block_num_edges 100000 \
    -num_nodes 80513 \
    -init_learning_rate 0.025 \
    -dict_file data/flickr/degree.txt \
    -negative_num 5 \
    -debug 0 

