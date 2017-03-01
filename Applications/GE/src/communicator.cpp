#include <cassert>
#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include "communicator.h"

namespace graphembedding {

Communicator::Communicator(const Option* o) : option_(o) {
    W_IE_table_ = NULL;
    W_OE_table_ = NULL;
    PrepareTable();
}

Communicator::~Communicator() {
    ClearTable();
}

void Communicator::PrepareTable() {
    if (W_IE_table_ || W_OE_table_) return;
    multiverso::Log::Info("Rank %d preparing table\n", multiverso::MV_Rank());

    integer row = option_->num_nodes;
    int col = option_->embedding_size;
    W_IE_table_ = multiverso::MV_CreateTable(
        MemSaveMatrixTableOption<real>(row, col, (real)-.5 / col, (real).5 / col));

    if (option_->algo_type == AlgorithmType::Line) {
        W_OE_table_ = multiverso::MV_CreateTable(
            MemSaveMatrixTableOption<real>(row, col, (real)-.5 / col, (real).5 / col));
    }
    multiverso::Log::Info("Rank %d preparing table finished\n", multiverso::MV_Rank());
}

void Communicator::ClearTable() {
    if (W_IE_table_ != NULL) { delete W_IE_table_; W_IE_table_ = NULL; }
    if (W_OE_table_ != NULL) { delete W_OE_table_; W_OE_table_ = NULL; }
}

void Communicator::RequestParameter(DataBlock* db) {
    if (db->input_nodes.size() != 0) {
        assert(W_IE_table_ != NULL);
        size_t input_size = db->input_nodes.size();
        std::vector<integer> input_ids(db->input_nodes.begin(), db->input_nodes.end());
        std::vector<real*> input_vec(input_size, NULL);
        for (auto i = 0; i < input_size; ++ i) {
            input_vec[i] = db->GetWeightIE(input_ids[i]);
        }
        W_IE_table_->Get(input_ids, input_vec);  
    }

    if (db->output_nodes.size() != 0) {
        assert(W_OE_table_ != NULL);
        size_t output_size = db->output_nodes.size();
        std::vector<integer> output_ids(db->output_nodes.begin(), db->output_nodes.end());
        std::vector<real*> output_vec(output_size, NULL);
        for (auto i = 0; i < output_size; ++ i) {
            output_vec[i] = db->GetWeightOE(output_ids[i]);
        }
        W_OE_table_->Get(output_ids, output_vec);  
    }
}

void Communicator::AddParameter(DataBlock* db) {
    if (db->input_nodes.size() != 0) {
        size_t input_size = db->input_nodes.size();
        std::vector<integer> input_ids(db->input_nodes.begin(), db->input_nodes.end());
        std::vector<real*> input_vec(input_size, NULL);
        for (auto i = 0; i < input_size; ++ i) {
            input_vec[i] = db->GetWeightIE(input_ids[i]);
        }
        W_IE_table_->Add(input_ids, input_vec);  
    }

    if (db->output_nodes.size() != 0) {
        assert(W_OE_table_ != NULL);
        size_t output_size = db->output_nodes.size();
        std::vector<integer> output_ids(db->output_nodes.begin(), db->output_nodes.end());
        std::vector<real*> output_vec(output_size, NULL);
        for (auto i = 0; i < output_size; ++ i) {
            output_vec[i] = db->GetWeightOE(output_ids[i]);
        }
        W_OE_table_->Add(output_ids, output_vec);  
    }
}

void Communicator::RequestParameterWI(
        const std::vector<integer>& nodes, 
        const std::vector<real*>& vecs) {
    assert(W_IE_table_ != NULL);
    W_IE_table_->Get(nodes, vecs);
}

}
