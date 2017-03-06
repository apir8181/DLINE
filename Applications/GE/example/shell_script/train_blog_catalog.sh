
mpirun -hostfile ./host.txt -bind-to none ../../../build/Applications/GE/graphembedding \
    -graph_part_file ./data/blog_catalog/edges.txt \
    -output_file ./data/blog_catalog/vector.txt \
    -rule_file ./rule.txt \
    -embedding_size 100 \
    -sample_edges_millions 120 \
    -block_num_edges 10000 \
    -num_nodes 10312 \
    -init_learning_rate 0.025 \
    -dict_file ./data/blog_catalog/degree.txt \
    -negative_num 5 \
    -server_threads 7 \
    -debug 0

