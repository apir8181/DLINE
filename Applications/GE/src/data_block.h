#ifndef GE_DATA_BLOCK_H
#define GE_DATA_BLOCK_H

#include <random>
#include <vector>
#include "constant.h"
#include "dictionary.h"
#include "util.h"

namespace graphembedding {

struct Edge {
    integer src, dst;
    real weight;
};

class DataBlock {
public:
    Edge* edges;
    std::vector<integer> negatives;
    std::vector<integer> input_nodes, output_nodes;

    DataBlock(Edge* edges, size_t size, const Option* option);

    ~DataBlock();

    size_t Size();

    void Use2OrderWithNegatives(Dictionary* dict, std::mt19937_64 &gen);
    
    void AllocParameters();

    void ClearParameters();

    void MakeParametersCopy();

    void MakeParametersDiff();

    real* GetWeightIE(integer idx);
    real* GetWeightOE(integer idx);

private:
    const Option* option_;
    const size_t size_;
    real* W_IE_;
    real* W_OE_;
    real* W_IE_copy_;
    real* W_OE_copy_;
};

}

#endif
