#include <cassert>
#include <cstdio>
#include <sstream>
#include <string>
#include <cmath>
#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include "model.h"

namespace graphembedding {

Model::Model(Option* option) : option_(option) {
    table_ = NULL;
    dict_ = NULL;
    graph_partition_ = NULL;
}

Model::~Model() {
    if (table_ != NULL) delete table_;
    if (dict_ != NULL) delete dict_;
    if (graph_partition_ != NULL) delete graph_partition_;
    multiverso::MV_ShutDown();
}

void Model::Init() {
    if (option_->debug) {
        multiverso::Log::ResetLogLevel(multiverso::LogLevel::Debug);
    }

    int argc = 1;
    multiverso::MV_Init(&argc, NULL);
    multiverso::MV_Barrier();
    rank_ = multiverso::MV_Rank();
    worker_id_ = multiverso::MV_WorkerId();
    server_id_ = multiverso::MV_ServerId();
    multiverso::Log::Info("MV Rank %d multiverso initialized\n", rank_);

    option_->sample_edges /= multiverso::MV_NumWorkers();

    integer row = option_->num_nodes;
    int col = option_->embedding_size;
    int num_threads = option_->server_threads;
    table_ = multiverso::MV_CreateTable(ColumnMatrixTableOption<real>(
                row, col, (real)-.5 / col, (real).5 / col, num_threads));
    if (worker_id_ != -1) {
        graph_partition_ = new (std::nothrow)GraphPartition(option_);
        assert(graph_partition_ != NULL);
        multiverso::Log::Info("MV Rank %d (Worker %d) preprocessed graph partition\n", 
            rank_, worker_id_);

        dict_ = new (std::nothrow)Dictionary(option_);
        assert(dict_ != NULL);
        multiverso::Log::Info("MV Rank %d (Worker %d) opened dictionary\n", 
            rank_, worker_id_);
    }
    multiverso::MV_Barrier();
}

void Model::Train() {
    integerL edge_processed = 0, block_processed = 0;
    while (edge_processed < option_->sample_edges && worker_id_ != -1) {
        PRINT_CLOCK_BEGIN(load);
        std::vector<Edge> edges = graph_partition_->ReadDataBlock();
        int edges_readed = edges.size();
        assert(edges_readed != 0);
        PRINT_CLOCK_END(load, "load");

        PRINT_CLOCK_BEGIN(dotprod);
        DotProdParam* dotprod_param = GetDotProdParam(edges, edges_readed);
        DotProdResult* dotprod_result = table_->DotProd(dotprod_param);
        PRINT_CLOCK_END(dotprod, "dotprod");

        PRINT_CLOCK_BEGIN(adjust);
        real lr = option_->init_learning_rate; 
        real loss;
        AdjustParam* adjust_param = GetAdjustParam(edges, lr,
            dotprod_param, dotprod_result, loss);
        table_->Adjust(adjust_param);
        PRINT_CLOCK_END(adjust, "adjust");

        delete dotprod_param;
        delete dotprod_result;
        delete adjust_param;

        edge_processed += edges_readed;
        block_processed += 1;

        real progress = std::min(1.0f, (real)edge_processed / option_->sample_edges);
        multiverso::Log::Info("Rank %d (Worker %d) Iter %d, loss %f, progress %f\n",
            rank_, worker_id_, block_processed, loss, progress); 
    }
    
    multiverso::Log::Info("Rank %d train finished\n", multiverso::MV_Rank());
    multiverso::MV_Barrier();
}

void Model::Save() {
    /*
    if (multiverso::MV_ServerId() == -1) return;
    
    multiverso::Log::Info("Rank %d saving parameters\n", multiverso::MV_Rank());
    integer row_offset, size;
    size = option_->num_nodes / multiverso::MV_NumServers();
    int server_id = multiverso::MV_ServerId();
    if (size > 0) {
        row_offset = size * server_id;
        if (server_id == multiverso::MV_NumServers() - 1) {
            size = option_->num_nodes - row_offset;
        }
    } else {
        size = server_id < option_->num_nodes ? 1 : 0;
        row_offset = server_id;
    }

    std::stringstream ss;
    ss << option_->output_file << "_part_" << multiverso::MV_Rank();
    std::ofstream fs(ss.str());
    if (!fs.is_open()) {
        multiverso::Log::Fatal("Rank %d can't open output file %s\n",
            multiverso::MV_Rank(), option_->output_file);
    }

    const integer BATCH_SIZE = 100000;
    real* buffer = new (std::nothrow)real[BATCH_SIZE * option_->embedding_size];
    assert(buffer != NULL);
    std::vector<integer> nodes;
    std::vector<real*> vecs;
    for (integer i = 0; i < size; i += BATCH_SIZE) {
        integer remain_size = i + BATCH_SIZE <= size ? BATCH_SIZE : size - i;
        // request local parameter
        nodes.resize(remain_size);
        vecs.resize(remain_size);
        for (integer j = 0; j < remain_size; ++ j) {
            nodes[j] = row_offset + i + j;
            vecs[j] = &buffer[j * option_->embedding_size];
        }
        communicator_->RequestParameterWI(nodes, vecs);
        // save local parameter
        for (integer j = 0; j < remain_size; ++ j) {
            fs << row_offset + i + j;
            real* x = vecs[i + j];
            for (int k = 0; k < option_->embedding_size; ++ k) {
                fs << ' ' << x[k];
            }
            fs << '\n';
        }
    }
    delete buffer;
    fs.close();
    */
}

DotProdParam* Model::GetDotProdParam(std::vector<Edge>& edges, int size) {
    DotProdParam* param = new (std::nothrow)DotProdParam();
    assert(param != NULL);
    for (int i = 0; i < size; ++ i) {
        const Edge& edge = edges[i];
        param->src.push_back(edge.src);
        param->dst.push_back(edge.dst);
        for (int j = 0; j < option_->negative_num; ++ j) {
            param->src.push_back(edge.src);
            integer neg = edge.dst;
            while (neg == edge.dst) {
                neg = dict_->Sample(gen_);
            }
            param->dst.push_back(neg);
        }
    }
    return param;
}

AdjustParam* Model::GetAdjustParam(std::vector<Edge>& edges, real lr, 
        DotProdParam* dotprod_param, DotProdResult* dotprod_result, real& loss) {
    assert(dotprod_param->src.size() == dotprod_result->scale.size());
    assert(dotprod_param->src.size() / (1 + option_->negative_num) == edges.size());
    AdjustParam* param = new (std::nothrow)AdjustParam();
    assert(param != NULL);
    
    int num_edges = edges.size();
    real total_loss = 0;
    for (size_t i = 0; i < num_edges; ++ i) {
        for (int j = 0; j < 1 + option_->negative_num; ++ j) {
            size_t idx = i * (1 + option_->negative_num) + j;
            real ip = dotprod_result->scale[idx];
            if (ip >= 10) ip = 10;
            if (ip <= -10) ip = -10;
            real sigmoid = 1 / (1 + exp(-ip));
            real weight = edges[i].weight;
            total_loss += j == 0 ? -weight * log(sigmoid) : -weight * log(1 - sigmoid);
            real scale = lr * weight * ((j == 0) - sigmoid);
            param->src.push_back(dotprod_param->src[idx]);
            param->dst.push_back(dotprod_param->dst[idx]);
            param->scale.push_back(scale);
        }
    }
    loss = total_loss / num_edges / (1 + option_->negative_num);

    return param;
}

}
