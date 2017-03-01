
#include "data_block.h"
#include "util.h"
#include <cassert>
#include <cstring>
#include <unordered_set>
#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include <algorithm>

namespace graphembedding {

DataBlock::DataBlock(Edge* es, size_t size, const Option* option) : 
        option_(option), size_(size) {
    edges = es;
    W_IE_ = W_IE_copy_ = NULL;
    W_OE_ = W_OE_copy_ = NULL;
}

DataBlock::~DataBlock() {
    ClearParameters();
    if (edges != NULL) delete[] edges;
}

size_t DataBlock::Size() {
    return size_;
}

void DataBlock::Use2OrderWithNegatives(Dictionary* dict, std::mt19937_64 &gen) {
    size_t N = size_, K = option_->negative_num;
    negatives.clear();
    input_nodes.clear();
    output_nodes.clear();

    // find input and output nodes
    for (auto i = 0; i < N; ++ i) {
        input_nodes.push_back(edges[i].src);
    }
    SortEraseDuplicate(input_nodes);

    // find negative pools
    std::vector<integer> negative_pools;
    size_t input_size = input_nodes.size();
    for (auto i = 0; i < input_size * K; ++ i) {
        negative_pools.push_back(dict->Sample(gen));
    }
    SortEraseDuplicate(negative_pools);

    // find negatives
    size_t pools_size = negative_pools.size(); 
    std::uniform_int_distribution<integer> dist(0, pools_size - 1);
    for (auto i = 0; i < N; ++ i) {
        output_nodes.push_back(edges[i].dst);
        for (int j = 0; j < K; ++ j) {
            integer neg = negative_pools[dist(gen)];
            while (neg == edges[i].dst) neg = negative_pools[dist(gen)];
            negatives.push_back(neg);
            output_nodes.push_back(neg);
        }
    }
    SortEraseDuplicate(output_nodes);

    /*
    // fill in negatives and input output nodes
    for (auto i = 0; i < N; ++ i) {
        const Edge& edge = edges[i];
        input_nodes.push_back(edge.src);
        output_nodes.push_back(edge.dst);

        integer neg;
        for (auto j = 0; j < K; ++ j) {
            neg = edge.dst;
            while (neg == edge.dst) {
                neg = dict->Sample(gen);
            }
            negatives.push_back(neg);
            output_nodes.push_back(neg);
        }
    }
    SortEraseDuplicate(input_nodes);
    SortEraseDuplicate(output_nodes);
    */
}

void DataBlock::AllocParameters() {
    ClearParameters();

    int d = option_->embedding_size;
   
    size_t W_IE_size = input_nodes.size() * d + 1;
    W_IE_ = (real*)calloc(W_IE_size, sizeof(real));
    W_IE_copy_ = (real*)calloc(W_IE_size, sizeof(real));
    assert(W_IE_ != NULL && W_IE_copy_ != NULL);

    if (option_->algo_type == AlgorithmType::Line) {
        size_t W_OE_size = output_nodes.size() * d + 1;
        W_OE_ = (real*)calloc(W_OE_size, sizeof(real));
        W_OE_copy_ = (real*)calloc(W_OE_size, sizeof(real));
        assert(W_OE_ != NULL && W_OE_copy_ != NULL);
    }
}

void DataBlock::ClearParameters() {
    if (W_IE_ != NULL) { free(W_IE_); W_IE_ = NULL; }
    if (W_OE_ != NULL) { free(W_OE_); W_OE_ = NULL; }
    if (W_IE_copy_ != NULL) { free(W_IE_copy_); W_IE_copy_ = NULL; }
    if (W_OE_copy_ != NULL) { free(W_OE_copy_); W_OE_copy_ = NULL; }
}

real* DataBlock::GetWeightIE(integer node) {
    integerL idx = BinarySearch(input_nodes, node);
    return &W_IE_[idx * option_->embedding_size];
}

real* DataBlock::GetWeightOE(integer node) {
    integerL idx = BinarySearch(output_nodes, node);
    return &W_OE_[idx * option_->embedding_size];
}

void DataBlock::MakeParametersCopy() {
    int d = option_->embedding_size;
    
    assert(W_IE_ != NULL && W_IE_copy_ != NULL);
    size_t W_IE_size = input_nodes.size() * d + 1;
    memcpy(W_IE_copy_, W_IE_, W_IE_size * sizeof(real));

    if (option_->algo_type == AlgorithmType::Line) {
        assert(W_OE_ != NULL && W_OE_copy_ != NULL);
        size_t W_OE_size = output_nodes.size() * d + 1;
        memcpy(W_OE_copy_, W_OE_, W_OE_size * sizeof(real));
    }
}

void DataBlock::MakeParametersDiff() {
    int d = option_->embedding_size;

    assert(W_IE_ != NULL && W_IE_copy_ != NULL);
    size_t W_IE_size = input_nodes.size() * d + 1;
    for (auto i = 0; i < W_IE_size; ++ i) {
        W_IE_[i] = (W_IE_[i] - W_IE_copy_[i]);
    }

    if (option_->algo_type == AlgorithmType::Line) {
        assert(W_OE_ != NULL && W_OE_copy_ != NULL);
        size_t W_OE_size = output_nodes.size() * d + 1;
        for (auto i = 0; i < W_OE_size; ++ i) {
            W_OE_[i] = (W_OE_[i] - W_OE_copy_[i]);
        }
    }
}

}
