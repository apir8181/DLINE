#include <cassert>
#include <algorithm>
#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include "graph_partition.h"

namespace graphembedding {

GraphPartition::GraphPartition(const Option* o) : option_(o) {
    file_path_ = option_->graph_part_file;
    rank_ = multiverso::MV_Rank();
    worker_id_ = multiverso::MV_WorkerId();
    //file_path_ += "_part_" + std::to_string(worker_id_);

    pFILE_ = fopen(file_path_.c_str(), "r");
    if (pFILE_ == NULL) {
        multiverso::Log::Fatal("Rank %d (Worker %d) can't open file %s\n",
            rank_, worker_id_, file_path_.c_str());
    } else {
        multiverso::Log::Info("Rank %d (Worker %d) open file %s\n",
            rank_, worker_id_, file_path_.c_str());
    }

    edges_in_file_ = 0;
    multiverso::Log::Info("Rank %d (Worker %d) counting number of edges\n",
        rank_, worker_id_);
    integer a, b;
    real w;
    while (fscanf(pFILE_, "%d %d %f", &a, &b, &w) != EOF) {
        edges_in_file_ ++;
    }
    multiverso::Log::Info("Rank %d (Worker %d) contains %lld edges\n",
        rank_, worker_id_, edges_in_file_);
    assert(edges_in_file_ != 0);
    ResetStream();

    edges_remained_ = option_->sample_edges;
}

GraphPartition::~GraphPartition() {
    if (pFILE_ != NULL) fclose(pFILE_);
}

void GraphPartition::ResetStream() {
    fclose(pFILE_);
    pFILE_ = fopen(file_path_.c_str(), "r");
    if (pFILE_ == NULL) {
        multiverso::Log::Fatal("Rank %d (Worker %d) can't open file %s\n",
            rank_, worker_id_, file_path_.c_str());
    } 
}

std::vector<Edge> GraphPartition::ReadDataBlock() {
    integerL edges_readed = std::min(option_->block_num_edges, edges_remained_);
    std::vector<Edge> edges(edges_readed);
    if (edges_readed != 0) {
        for (auto i = 0; i < edges_readed; ++ i) {
            while (fscanf(pFILE_, "%d %d %f", 
                    &edges[i].src, &edges[i].dst, 
                    &edges[i].weight) == EOF) {
                ResetStream();
            } 
        }
        edges_remained_ -= edges_readed;
    }
    return edges;
}

}
