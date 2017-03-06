
mpirun -hostfile ./host.txt -bind-to none ../../../build/Applications/GE/graphembedding \
    -graph_part_file ./data/youtube/edges.txt \
    -rule_file ./rule.txt \
    -output_file ./vector.txt \
    -embedding_size 100 \
    -sample_edges_millions 240 \
    -block_num_edges 10000 \
    -num_nodes 1138499 \
    -init_learning_rate 0.025 \
    -dict_file ./data/youtube/degree.txt \
    -negative_num 5 \
    -debug 0 \
    -display_iter 10\
    -server_threads 20 

