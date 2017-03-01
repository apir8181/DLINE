#include <cassert>
#include <vector>
#include <string>
#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include "graph_partition.h"

namespace graphembedding {

GraphPartition::GraphPartition(const Option* o) : option_(o) {
    file_path_ = option_->graph_part_file;
    file_path_ += "_part_" + std::to_string(multiverso::MV_Rank());

    pFILE_ = fopen(file_path_.c_str(), "r");
    if (pFILE_ == NULL) {
        multiverso::Log::Fatal("Rank %d can't open file %s\n",
            multiverso::MV_Rank(), file_path_.c_str());
    } else {
        multiverso::Log::Info("Rank %d open file %s\n",
            multiverso::MV_Rank(), file_path_.c_str());
    }

    edges_in_file_ = 0;
    multiverso::Log::Info("Rank %d counting number of edges\n",
        multiverso::MV_Rank());
    integer a, b;
    real w;
    while (fscanf(pFILE_, "%d %d %f", &a, &b, &w) != EOF) {
        edges_in_file_ ++;
        total_weight_ += w;
    }
    multiverso::Log::Info("Rank %d contains %lld edges with total weight %f\n", 
        multiverso::MV_Rank(), edges_in_file_, total_weight_);
    assert(edges_in_file_ != 0);
    ResetStream();

    edges_remained_ = option_->sample_edges;
}

GraphPartition::~GraphPartition() {
    if (pFILE_ != NULL) fclose(pFILE_);
}

realL GraphPartition::TotalWeight() {
    return total_weight_;
}

void GraphPartition::ResetStream() {
    fclose(pFILE_);
    pFILE_ = fopen(file_path_.c_str(), "r");
    if (pFILE_ == NULL) {
        multiverso::Log::Fatal("Rank %d can't open file %s\n",
            multiverso::MV_Rank(), file_path_.c_str());
    } 
}

bool GraphPartition::ReadDataBlock(DataBlock*& db) {
    integerL edges_readed = std::min(option_->block_num_edges, edges_remained_);
    if (edges_readed == 0) {
        db = NULL;
        return false;
    } else {
        Edge* edges = new (std::nothrow)Edge[edges_readed]();
        assert(edges != NULL);
        for (auto i = 0; i < edges_readed; ++ i) {
            while (fscanf(pFILE_, "%d %d %f", 
                    &edges[i].src, &edges[i].dst, 
                    &edges[i].weight) == EOF) {
                ResetStream();
            } 
        }
        db = new DataBlock(edges, edges_readed, option_);
        edges_remained_ -= edges_readed;
        return true;
    }
}

}
