#include <cmath>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <algorithm>
#include <multiverso/util/log.h>
#include "util.h"

namespace graphembedding {

Option::Option() {
    graph_part_file = NULL;
    output_file = NULL;
    algo_type = AlgorithmType::Line;
    debug = false;
    embedding_size = 100;
    num_nodes = 1e6;
    sample_edges = 1e6;
    block_num_edges = 1e5;
    compute_threads = 1;
    preprocess_threads = 1;
    init_learning_rate = (real).025;
    dict_file = NULL;
    negative_num = 5;
}

void Option::ParseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; i += 2) { 
        if (strcmp(argv[i], "-graph_part_file") == 0) graph_part_file = argv[i + 1];
        if (strcmp(argv[i], "-output_file") == 0) output_file = argv[i + 1];
        if (strcmp(argv[i], "-algo_type") == 0) {
            if (strcmp(argv[i + 1], "line") == 0) algo_type = AlgorithmType::Line;
            else if (strcmp(argv[i + 1], "gf") == 0) algo_type = AlgorithmType::GF;
        }
        if (strcmp(argv[i], "-debug") == 0) debug = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-embedding_size") == 0) embedding_size = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-compute_threads") == 0) compute_threads = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-preprocess_threads") == 0) preprocess_threads = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-num_nodes") == 0) num_nodes = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-sample_edges_millions") == 0) {
            sample_edges = atoll(argv[i + 1]);
            sample_edges *= 1e6;
        }
        if (strcmp(argv[i], "-block_num_edges") == 0) block_num_edges = atoll(argv[i + 1]);
        if (strcmp(argv[i], "-init_learning_rate") == 0) init_learning_rate = atof(argv[i + 1]);
        if (strcmp(argv[i], "-dict_file") == 0) dict_file = argv[i + 1];
        if (strcmp(argv[i], "-negative_num") == 0) negative_num = atoi(argv[i + 1]);
    }
}

void Option::PrintUsage() {
    puts("Usage:");
    puts("-graph_part_file: local path for a graph partition");
    puts("-output_file: local path for embedding matrix");
    puts("-algo_type: embedding algorithm type, [ge, line1, line2]");
    puts("-debug: open debug log when setting this nonzero");
    puts("-embedding_size: embedding size");
    puts("-compute_threads: threads to compute forward and backward computation in NN");
    puts("-preprocess_threads: threads to preprocess loaded data");
    puts("-num_nodes: number of nodes in the graph");
    puts("-sample_edges_millions: number of edges sample totally (in millions) in training process");
    puts("-block_num_edges: number of edges in each training datablock");
    puts("-init_learning_rate: initialized learning rate");
    puts("-dict_file: dictionary file for negative sampling. Only used in line2");
    puts("-negative_num: negative sampling size for each edge. Only used in line2");
}

void Option::PrintArgs() {
    multiverso::Log::Info("Arguments:\n");
    multiverso::Log::Info("\tgraph_part_file: %s\n", graph_part_file);
    multiverso::Log::Info("\toutput_file: %s\n", output_file);
    switch (algo_type) {
        case AlgorithmType::GF: 
            multiverso::Log::Info("\talgo_type: Laplacian Graph Factorization\n");
            break;
        case AlgorithmType::Line: 
            multiverso::Log::Info("\talgo_type: Line First Order\n");
            break;
        default:
            multiverso::Log::Fatal("\talgo_type: EMPTY\n");
            break;
    }
    multiverso::Log::Info("\tdebug: %d\n", debug);
    multiverso::Log::Info("\tembedding_size: %d\n", embedding_size);
    multiverso::Log::Info("\tcompute_threads: %d\n", compute_threads);
    multiverso::Log::Info("\tpreprocess_threads: %d\n", preprocess_threads);
    multiverso::Log::Info("\tnum_nodes: %d\n", num_nodes);
    multiverso::Log::Info("\tsample_edges: %lld\n", sample_edges);
    multiverso::Log::Info("\tblock_num_edges: %d\n", block_num_edges);
    multiverso::Log::Info("\tinit_learning_rate: %f\n", init_learning_rate);
    multiverso::Log::Info("\tdict_file: %s\n", dict_file);
    multiverso::Log::Info("\tnegative_num: %d\n", negative_num);
}

AliasMethod::AliasMethod(const std::vector<real>& weight) {
    N_ = weight.size();
    assert(N_ != 0);
    prob_.resize(N_, 0);
    shared_.resize(N_, 0);

    realL sum_weight = 0;
    for (auto i = 0; i < N_; ++ i) sum_weight += weight[i];
    
    std::vector<integer> smaller(N_), larger(N_);
    size_t smaller_count = 0, larger_count = 0;
    for (auto i = 0; i < N_; ++ i) {
        real prob = real(N_ * weight[i] / sum_weight);
        prob_[i] = prob;
        shared_[i] = i;
        if (prob < (real)1.0) {
            smaller[smaller_count++] = i;
        } else {
            larger[larger_count++] = i;
        }
    }

    integer s, l;
    while (smaller_count > 0 && larger_count > 0) {
        s = smaller[--smaller_count];
        l = larger[--larger_count];

        shared_[s] = l;
        prob_[l] -= (real)(1.0) - prob_[s];
        if (prob_[l] < (real)1.0) {
            smaller[smaller_count++] = l;
        } else {
            larger[larger_count++] = l;
        }
    }
}

AliasMethod::~AliasMethod() {
    prob_.clear();
    shared_.clear();
}

template<typename T>
integerL BinarySearch(const std::vector<T>& array, T target) {
    integerL start = 0, end = array.size() - 1, mid;
    while (start <= end) {
        mid = (start + end) >> 1;
        if (array[mid] == target) return mid;
        else if (array[mid] < target) start = mid + 1;
        else end = mid - 1;
    }
    return -1;
}

template integerL BinarySearch<integer>(const std::vector<integer>& array, integer target);

template<typename T>
void SortEraseDuplicate(std::vector<T>& array) {
    std::unordered_set<T> S(array.begin(), array.end());
    array.assign(S.begin(), S.end());
    std::sort(array.begin(), array.end());
}

template void SortEraseDuplicate<integer>(std::vector<integer>& array);

}

