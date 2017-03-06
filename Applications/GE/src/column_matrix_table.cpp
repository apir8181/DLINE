#include <random>
#include <cassert>
#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include <multiverso/updater/updater.h>
#include <multiverso/io/io.h>
#include "column_matrix_table.h"
#include "util.h"

namespace graphembedding {

enum class Op { GET, DOTPROD, ADJUST };

template<typename T>
ColumnMatrixWorkerTable<T>::ColumnMatrixWorkerTable(
        const ColumnMatrixTableOption<T>& option) : WorkerTable() {
    num_cols_ = option.num_cols;
    num_servers_ = multiverso::MV_NumServers();
    rank_ = multiverso::MV_Rank();
    worker_id_ = multiverso::MV_WorkerId();
}

template<typename T>
ColumnMatrixWorkerTable<T>::~ColumnMatrixWorkerTable() {}

template<typename T>
DotProdResult* ColumnMatrixWorkerTable<T>::DotProd(DotProdParam* param) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(param->src.size() == param->dst.size());
    int num_edges = param->src.size();

    Blob blob(sizeof(integer) * 2 + num_edges * sizeof(integer) * 2);
    char* data = blob.data();
    dotprod_result_ = new DotProdResult();
    dotprod_result_->scale.resize(num_edges, 0);

    reinterpret_cast<integer*>(data)[0] = (int)Op::DOTPROD;
    data += sizeof(integer);

    reinterpret_cast<integer*>(data)[0] = num_edges;
    data += sizeof(integer);

    memcpy(data, param->src.data(), num_edges * sizeof(integer));
    data += num_edges * sizeof(integer);

    memcpy(data, param->dst.data(), num_edges * sizeof(integer));

    WorkerTable::Get(blob, NULL);
    multiverso::Log::Debug("[DotProd] Rank %d (Worker = %d), num_edges = %d\n",
        rank_, worker_id_, num_edges);
    return dotprod_result_;
}

template<typename T>
void ColumnMatrixWorkerTable<T>::Adjust(AdjustParam* param) {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(param->src.size() == param->dst.size() && 
            param->src.size() == param->scale.size());
    int num_edges = param->src.size();
    int num_src_unique = param->src_unique.size();
    int num_dst_unique = param->dst_unique.size();

    size_t blob_size = sizeof(integer) * 4;
    blob_size += (2 * sizeof(integer) + sizeof(real)) * num_edges;
    blob_size += sizeof(integer) * (num_src_unique + num_dst_unique);
    Blob blob(blob_size);
    char* data = blob.data();

    reinterpret_cast<integer*>(data)[0] = (int)Op::ADJUST;
    data += sizeof(integer);

    reinterpret_cast<integer*>(data)[0] = num_edges;
    data += sizeof(integer);

    reinterpret_cast<integer*>(data)[0] = num_src_unique;
    data += sizeof(integer);

    reinterpret_cast<integer*>(data)[0] = num_dst_unique;
    data += sizeof(integer);

    memcpy(data, param->src.data(), num_edges * sizeof(integer));
    data += num_edges * sizeof(integer);

    memcpy(data, param->dst.data(), num_edges * sizeof(integer));
    data += num_edges * sizeof(integer);

    memcpy(data, param->scale.data(), num_edges * sizeof(real));
    data += num_edges * sizeof(real);

    memcpy(data, param->src_unique.data(), num_src_unique * sizeof(integer));
    data += num_src_unique * sizeof(integer);

    memcpy(data, param->dst_unique.data(), num_dst_unique * sizeof(integer));
    data += num_dst_unique * sizeof(integer);

    WorkerTable::Get(blob, NULL);
    multiverso::Log::Debug("[Adjust] Rank %d (Worker = %d), num_edges = %d\n",
        rank_, worker_id_, num_edges);
}

template<typename T>
GetResult* ColumnMatrixWorkerTable<T>::Get(GetParam* param) {
    std::lock_guard<std::mutex> lock(mutex_);
    int num_nodes = param->src.size();

    get_result_ = new GetResult();
    get_result_->W.resize(size_t(num_nodes) * num_cols_);

    Blob blob( sizeof(integer) * 2 + num_nodes * sizeof(integer));
    char* data = blob.data();

    reinterpret_cast<integer*>(data)[0] = (int)Op::GET;
    data += sizeof(integer);

    reinterpret_cast<integer*>(data)[0] = num_nodes;
    data += sizeof(integer);

    memcpy(data, param->src.data(), num_nodes * sizeof(integer));
    
    WorkerTable::Get(blob, NULL);
    multiverso::Log::Debug("[Get] Rank %d (Worker = %d), num_nodes = %d\n",
        rank_, worker_id_, num_nodes);

    return get_result_;
}

template<typename T>
int ColumnMatrixWorkerTable<T>::Partition(
        const std::vector<Blob>& kv, multiverso::MsgType,
        std::unordered_map<int, std::vector<Blob> >* out) {
    for (auto i = 0; i < num_servers_; i++) {     
        int rank = multiverso::MV_ServerIdToRank(i);
        std::vector<Blob>& vec = (*out)[rank];
        vec.push_back(kv[0]);
    }
    return static_cast<int>(out->size());
}

template <typename T>
void ColumnMatrixWorkerTable<T>::ProcessReplyGet(std::vector<Blob>& reply_data) {
    assert(reply_data.size() == 1);
    Blob blob = reply_data[0];
    char* data = blob.data();

    int type = reinterpret_cast<int*>(data)[0];
    data += sizeof(integer);

    int num_edges = reinterpret_cast<int*>(data)[0];
    data += sizeof(integer);

    if (type == (int)Op::DOTPROD) {
        real* scale = reinterpret_cast<real*>(data);
        for (int i = 0; i < num_edges; ++ i) dotprod_result_->scale[i] += scale[i]; 
        multiverso::Log::Debug("[ProcessDotProd] Rank %d (Worker %d), "
            "#num_edges = %lld\n", rank_, worker_id_, num_edges);
    } else if (type == (int)Op::ADJUST) {
        multiverso::Log::Debug("[ProcessAdjust] Rank %d (Worker %d), "
            "#num_edges = %lld\n", rank_, worker_id_, num_edges);
    } else if (type == (int)Op::GET) {
        int server_offset = reinterpret_cast<int*>(data)[0];
        data += sizeof(integer);

        int server_cols = reinterpret_cast<int*>(data)[0];
        data += sizeof(integer);

        real* W1 = reinterpret_cast<real*>(get_result_->W.data());
        real* W2 = reinterpret_cast<real*>(data);
        for (size_t i = 0; i < num_edges; ++ i) {
            size_t src_offset = i * server_cols;
            size_t dst_offset = i * num_cols_ + server_offset;
            memcpy(W1 + dst_offset, W2 + src_offset, server_cols * sizeof(real));
        }
        multiverso::Log::Debug("[ProcessGet] Rank %d (Worker %d), "
            "#num_nodes = %lld\n", rank_, worker_id_, num_edges);
    }
}

template <typename T>
ColumnMatrixServerTable<T>::ColumnMatrixServerTable(
        const ColumnMatrixTableOption<T>& option) : ServerTable() {
    num_rows_ = option.num_rows;
    num_cols_ = option.num_cols;
    num_servers_ = multiverso::MV_NumServers();
    rank_ = multiverso::MV_Rank();
    server_id_ = multiverso::MV_ServerId();
    num_threads_ = option.threads;
    assert(num_cols_ >= num_servers_);

    num_cols_local_ = num_cols_ / num_servers_;
    offset_ = num_cols_local_ * server_id_;
    if (server_id_ == num_servers_ - 1) {
        num_cols_local_ = num_cols_ - offset_;
    }

    // create storage data
    size_t total_size = size_t(num_rows_) * num_cols_local_;
    W_IN_.resize(total_size, 0);
    W_OUT_.resize(total_size, 0);
    DW_IN_.resize(total_size, 0);
    DW_OUT_.resize(total_size, 0);

    // initialization 
    std::mt19937 gen(997 + rank_);
    std::uniform_real_distribution<float> dis(option.min_val, option.max_val);
    for (auto i = 0; i < total_size; i ++) {
      W_IN_[i] = dis(gen);
      W_OUT_[i] = dis(gen);
    }

    multiverso::Log::Info("[Init] Rank %d (Server %d), type = ColumnMatrixTable,"
        " size = [%lld x %lld], local size = [%lld x %lld]\n",
        rank_, server_id_, num_rows_, num_cols_, num_rows_, num_cols_local_);
}

template<typename T>
ColumnMatrixServerTable<T>::~ColumnMatrixServerTable() {}

template<typename T>
void ColumnMatrixServerTable<T>::ProcessGet(
        const std::vector<Blob>& kv,
        std::vector<Blob>* result) {
    assert(kv.size() == 1);
    char* data = kv[0].data();

    int type = reinterpret_cast<integer*>(data)[0];
    data += sizeof(integer);

    int num_edges = reinterpret_cast<integer*>(data)[0];
    data += sizeof(integer);

    if (type == (int)Op::DOTPROD) {
        integer* src = reinterpret_cast<integer*>(data);
        data += num_edges * sizeof(integer);
        integer* dst = reinterpret_cast<integer*>(data);
        
        result->push_back(Blob(2 * sizeof(integer) + num_edges * sizeof(real)));
        char* result_data = result->at(0).data();

        reinterpret_cast<integer*>(result_data)[0] = (int)Op::DOTPROD;
        result_data += sizeof(integer);

        reinterpret_cast<integer*>(result_data)[0] = num_edges;
        result_data += sizeof(integer);

        real* scale = reinterpret_cast<real*>(result_data);
        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < num_edges; ++ i) {
            scale[i] = 0;
            size_t src_offset = size_t(src[i]) * num_cols_local_;
            size_t dst_offset = size_t(dst[i]) * num_cols_local_;
            for (int j = 0; j < num_cols_local_; ++ j) {
               scale[i] += W_IN_[src_offset + j] * W_OUT_[dst_offset + j];
            }
        }
        multiverso::Log::Debug("[ProcessDotProd] Rank %d (Server %d), #num_edges=%d\n",
            rank_, server_id_, num_edges);
    } else if (type == (int)Op::ADJUST) {
        int num_src_unique = reinterpret_cast<integer*>(data)[0];
        data += sizeof(int);

        int num_dst_unique = reinterpret_cast<integer*>(data)[0];
        data += sizeof(int);

        integer* src = reinterpret_cast<integer*>(data);
        data += num_edges * sizeof(integer);

        integer* dst = reinterpret_cast<integer*>(data);
        data += num_edges * sizeof(integer);

        real* scale = reinterpret_cast<real*>(data);
        data += num_edges * sizeof(real);

        integer* src_unique = reinterpret_cast<integer*>(data);
        data += num_src_unique * sizeof(integer);

        integer* dst_unique = reinterpret_cast<integer*>(data);
        data += num_dst_unique * sizeof(integer);

        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < num_edges; ++ i) {
            size_t src_offset = size_t(src[i]) * num_cols_local_;
            size_t dst_offset = size_t(dst[i]) * num_cols_local_;
            for (int j = 0; j < num_cols_local_; ++ j) {
                DW_IN_[src_offset + j] += scale[i] * W_OUT_[dst_offset + j];
                DW_OUT_[dst_offset + j] += scale[i] * W_IN_[src_offset + j];
            }
        }

        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < num_src_unique; ++ i) {
            size_t offset = src_unique[i] * size_t(num_cols_local_);
            for (int j = 0; j < num_cols_local_; ++ j) {
                W_IN_[offset + j] += DW_IN_[offset + j];
                DW_IN_[offset + j] = 0;
            }
        }

        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < num_dst_unique; ++ i) {
            size_t offset = dst_unique[i] * size_t(num_cols_local_);
            for (int j = 0; j < num_cols_local_; ++ j) {
                W_OUT_[offset + j] += DW_OUT_[offset + j];
                DW_OUT_[offset + j] = 0;
            }
        }

        result->push_back(Blob(2 * sizeof(integer)));
        void* result_data = result->at(0).data();
        reinterpret_cast<integer*>(result_data)[0] = (int)Op::ADJUST;
        reinterpret_cast<integer*>(result_data)[1] = num_edges;
        multiverso::Log::Debug("[ProcessAdjust] Rank %d (Server %d), #num_edges=%d\n",
            rank_, server_id_, num_edges);
    } else if (type == (int)Op::GET) {
        result->push_back(Blob(4 * sizeof(integer) + sizeof(real) * num_edges * num_cols_local_));
        char* result_data = result->at(0).data();
        
        reinterpret_cast<integer*>(result_data)[0] = (int)Op::GET;
        result_data += sizeof(integer);

        reinterpret_cast<integer*>(result_data)[0] = num_edges;
        result_data += sizeof(integer);

        reinterpret_cast<integer*>(result_data)[0] = offset_;
        result_data += sizeof(integer);

        reinterpret_cast<integer*>(result_data)[0] = num_cols_local_;
        result_data += sizeof(integer);

        integer* src = reinterpret_cast<integer*>(data);
        real* W_dst = reinterpret_cast<real*>(result_data);
        real* W_src = reinterpret_cast<real*>(W_IN_.data());
        #pragma omp parallel for num_threads(num_threads_)
        for (size_t i = 0; i < num_edges; ++ i) {
            size_t dst_offset = i * num_cols_local_;
            size_t src_offset = src[i] * num_cols_local_;
            memcpy(W_dst + dst_offset, W_src + src_offset, num_cols_local_ * sizeof(real));
        }
    }
}

template<typename T>
void ColumnMatrixServerTable<T>::ProcessAdd(const std::vector<Blob>& kv) {}

template<typename T>
void ColumnMatrixServerTable<T>::Store(multiverso::Stream* s) {}

template<typename T>
void ColumnMatrixServerTable<T>::Load(multiverso::Stream *s) {}

MV_INSTANTIATE_CLASS_WITH_REAL_TYPE(ColumnMatrixWorkerTable);
MV_INSTANTIATE_CLASS_WITH_REAL_TYPE(ColumnMatrixServerTable);

}
