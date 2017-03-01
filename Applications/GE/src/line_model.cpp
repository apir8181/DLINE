
#include "line_model.h"
#include <cassert>
#include <cmath>
#include <vector>

namespace graphembedding {

LineModel::LineModel(Option* option) : Model(option) {
    dictionary_ = new (std::nothrow)Dictionary(option);
    assert(dictionary_ != NULL);
}

LineModel::~LineModel() {
    if (dictionary_ != NULL) delete dictionary_;
}

void LineModel::ThreadPreprocess(DataBlock* db, int thread_idx) {
    static thread_local std::mt19937_64 gen(multiverso::MV_Rank());
    // disable sampling in this part
    size_t N = db->Size();
    assert(N != 0);

    // calculate cumlative sum
    std::vector<real> weight(N);
    for (auto i = 0; i < N; ++ i) weight[i] = db->edges[i].weight;

    AliasMethod alias(weight);
    Edge* edges = new (std::nothrow)Edge[N]();
    assert(edges != NULL);
    // sample edges using cumulative sum sampling method
    for (auto i = 0; i < N; ++ i) {
        edges[i] = db->edges[alias.Sample(gen)];
    }
    delete[] db->edges;
    db->edges = edges;
    db->Use2OrderWithNegatives(dictionary_, gen);
}

void LineModel::ThreadTrain(DataBlock* db, real lr, int thread_idx, void* args) {
    const int D = option_->embedding_size;
    const int K = option_->negative_num;
    real* input_vec;
    real* output_vec;
    real* input_err = new (std::nothrow)real[D]();
    real* pos_err = new (std::nothrow)real[D]();
    real* neg_err = new (std::nothrow)real[D * K]();
    assert(input_err != NULL && pos_err != NULL && neg_err != NULL);

    realL total_loss = 0, weight;
    size_t N = db->Size();
    size_t seen = 0;
    for (auto i = thread_idx; i < N; i += option_->compute_threads) {
        const Edge& edge = db->edges[i];
        memset(input_err, 0, D * sizeof(real));
        memset(pos_err, 0, D * sizeof(real));
        memset(neg_err, 0, D * K * sizeof(real));

        // Forward
        input_vec = db->GetWeightIE(edge.src);
        output_vec = db->GetWeightOE(edge.dst);
        total_loss += TrainEdge(input_vec, output_vec, 
                input_err, pos_err, true);
        
        for (int k = 0; k < K; ++ k) {
            output_vec = db->GetWeightOE(db->negatives[i * K + k]);
            total_loss += TrainEdge(input_vec, output_vec, 
                    input_err, neg_err + k * D, false); 
        }

        // Backward
        output_vec = db->GetWeightOE(edge.dst);
        Update(input_vec, input_err, lr);
        Update(output_vec, pos_err, lr);

        for (int k = 0; k < K; ++ k) {
            output_vec = db->GetWeightOE(db->negatives[i * K + k]);
            Update(output_vec, neg_err + k * D, lr);            
        }
        ++ seen;
    }

    delete[] input_err;
    delete[] pos_err;
    delete[] neg_err;

    if (seen != 0) {
        reinterpret_cast<real*>(args)[0] = total_loss / seen / (K + 1);
    }
}

real LineModel::TrainEdge(real* input_vec, real* output_vec, 
        real* input_err, real* output_err, bool label) {
    const int D = option_->embedding_size;

    real ip = 0;
    for (int i = 0; i < D; ++ i) {
        ip += input_vec[i] * output_vec[i];
    }
    if (ip >= 10) {
        ip = 10;
    } else if (ip <= -10) {
        ip = -10;
    }

    real sigmoid = 1 / (1 + exp(-ip));
    real err = (label - sigmoid);

    for (int i = 0; i < D; ++ i) {
        input_err[i] += err * output_vec[i];
        output_err[i] += err * input_vec[i];
    }

    real loss = label ? -log(sigmoid) : -log(1 - sigmoid);
    return loss;
}

void LineModel::Update(real* vec, real* err, real lr) {
    const int D = option_->embedding_size;
    for (int i = 0; i < D; ++ i) vec[i] += lr * err[i];
}


}
