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
    dict_file = NULL;
    rule_file = NULL;
    output_file = NULL;
    embedding_size = 100;
    negative_num = 5;
    num_nodes = 1e6;
    sample_edges = 1e6;
    block_num_edges = 1e5;
    init_learning_rate = (real).025;
    display_iter = 1;
    server_threads = 1;
    debug = false;
}

void Option::ParseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; i += 2) { 
        if (strcmp(argv[i], "-graph_part_file") == 0) graph_part_file = argv[i + 1];
        if (strcmp(argv[i], "-dict_file") == 0) dict_file = argv[i + 1];
        if (strcmp(argv[i], "-rule_file") == 0) rule_file = argv[i + 1];
        if (strcmp(argv[i], "-output_file") == 0) output_file = argv[i + 1];
        if (strcmp(argv[i], "-embedding_size") == 0) embedding_size = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-negative_num") == 0) negative_num = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-num_nodes") == 0) num_nodes = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-sample_edges_millions") == 0) {
            sample_edges = atoll(argv[i + 1]);
            sample_edges *= 1e6;
        }
        if (strcmp(argv[i], "-block_num_edges") == 0) block_num_edges = atoll(argv[i + 1]);
        if (strcmp(argv[i], "-init_learning_rate") == 0) init_learning_rate = atof(argv[i + 1]);
        if (strcmp(argv[i], "-display_iter") == 0) display_iter = atof(argv[i + 1]);
        if (strcmp(argv[i], "-server_threads") == 0) server_threads = atoi(argv[i + 1]);
        if (strcmp(argv[i], "-debug") == 0) debug = atoi(argv[i + 1]);
    }
}

void Option::PrintUsage() {
    puts("Usage:");
    puts("-graph_part_file: local path for a graph partition");
    puts("-dict_file: dictionary file for negative sampling.");
    puts("-rule_file: rule file for each machines,"); 
    puts("-output_file: local path for embedding matrix");
    puts("-embedding_size: embedding size");
    puts("-negative_num: negative sampling size for each edge.");
    puts("-num_nodes: number of nodes in the graph");
    puts("-sample_edges_millions: number of edges sample totally (in millions) in training process");
    puts("-block_num_edges: number of edges in each training datablock");
    puts("-init_learning_rate: initialized learning rate");
    puts("-display_iter: display iteration");
    puts("-server_threads: number of computation threads in server");
    puts("-debug: open debug log when setting this nonzero");
}

void Option::PrintArgs() {
    multiverso::Log::Info("Arguments:\n");
    multiverso::Log::Info("\tgraph_part_file: %s\n", graph_part_file);
    multiverso::Log::Info("\tdict_file: %s\n", dict_file);
    multiverso::Log::Info("\trule_file: %s\n", rule_file);
    multiverso::Log::Info("\toutput_file: %s\n", output_file);
    multiverso::Log::Info("\tembedding_size: %d\n", embedding_size);
    multiverso::Log::Info("\tnegative_num: %d\n", negative_num);
    multiverso::Log::Info("\tnum_nodes: %d\n", num_nodes);
    multiverso::Log::Info("\tsample_edges: %lld\n", sample_edges);
    multiverso::Log::Info("\tblock_num_edges: %d\n", block_num_edges);
    multiverso::Log::Info("\tinit_learning_rate: %f\n", init_learning_rate);
    multiverso::Log::Info("\tdisplay_iter: %f\n", display_iter);
    multiverso::Log::Info("\tserver_threads: %d\n", server_threads);
    multiverso::Log::Info("\tdebug: %d\n", debug);
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

