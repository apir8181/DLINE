#ifndef GE_MODEL_H
#define GE_MODEL_H

#include "host_rule.h"
#include "column_matrix_table.h"
#include "constant.h"
#include "dictionary.h"
#include "graph_partition.h"
#include "util.h"

namespace graphembedding {

class Model {
public:
    Model(Option* option);

    virtual ~Model();

    void Init();

    void Train();

    void Save();

    Option* option_;

private:
    DotProdParam* GetDotProdParam(std::vector<Edge>& edges, int size);
    AdjustParam* GetAdjustParam(std::vector<Edge>& edges, real lr, 
            DotProdParam* param, DotProdResult* result, real& loss);

    int rank_, worker_id_, server_id_;
    HostRule* host_rule_;
    ColumnMatrixWorkerTable<real>* table_;
    Dictionary* dict_;
    GraphPartition* graph_partition_;
    std::mt19937_64 gen_;
};

}

#endif
