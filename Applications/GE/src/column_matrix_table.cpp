#include <random>
#include <cassert>
#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include <multiverso/updater/updater.h>
#include <multiverso/io/io.h>
#include "column_matrix_table.h"
#include "util.h"

namespace graphembedding {

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
    Blob blob = param->ToBlob();
    dotprod_result_ = new DotProdResult();
    dotprod_result_->scale.resize(param->dst.size(), 0);
    WorkerTable::Get(blob, NULL);
    multiverso::Log::Debug("[DotProd] Rank %d (Worker = %d), num_edges=%d\n",
        rank_, worker_id_, param->dst.size());
    return dotprod_result_;
}

template<typename T>
void ColumnMatrixWorkerTable<T>::Adjust(AdjustParam* param) {
    std::lock_guard<std::mutex> lock(mutex_);
    Blob blob = param->ToBlob();
    WorkerTable::Get(blob, NULL);
    multiverso::Log::Debug("[Adjust] Rank %d (Worker = %d), num_edges=%d\n",
        rank_, worker_id_, param->dst.size());
}

template<typename T>
GetResult* ColumnMatrixWorkerTable<T>::Get(GetParam* param) {
    std::lock_guard<std::mutex> lock(mutex_);
    Blob blob = param->ToBlob(); 
    get_result_ = new GetResult();
    get_result_->W.resize(param->src.size() * num_cols_);
    WorkerTable::Get(blob, NULL);
    multiverso::Log::Debug("[Get] Rank %d (Worker = %d), num_nodes=%d\n",
        rank_, worker_id_, param->src.size());
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
    Op type = GetOpType(blob);
    if (type == Op::DOTPROD) {
        DotProdResult* result = DotProdResult::FromBlob(blob);
        int num_elems = result->scale.size();
        for (int i = 0; i < num_elems; ++ i) {
            dotprod_result_->scale[i] += result->scale[i]; 
        }
        delete result;
        multiverso::Log::Debug("[ProcessDotProd] Rank %d (Worker %d), "
            "#num_edges=%d\n", rank_, worker_id_, num_elems);
    } else if (type == Op::ADJUST) {
        multiverso::Log::Debug("[ProcessAdjust] Rank %d (Worker %d)\n", rank_, worker_id_);
    } else if (type == Op::GET) {
        GetResult* result = GetResult::FromBlob(blob);
        int num_elems = result->W.size();
        int num_rows = num_elems / result->cols_own;
        for (size_t i = 0; i < num_rows; ++ i) {
            size_t src_offset = i * result->cols_own;
            size_t dst_offset = i * num_cols_ + result->cols_offset;
            memcpy(get_result_->W.data() + dst_offset, result->W.data() + src_offset, 
                   result->cols_own * sizeof(real));
        }
        delete result;
        multiverso::Log::Debug("[ProcessGet] Rank %d (Worker %d), "
            "#num_nodes=%d\n", rank_, worker_id_, num_rows);
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
        " size = [%d x %d], local size = [%d x %d]\n",
        rank_, server_id_, num_rows_, num_cols_, num_rows_, num_cols_local_);
}

template<typename T>
ColumnMatrixServerTable<T>::~ColumnMatrixServerTable() {}

template<typename T>
void ColumnMatrixServerTable<T>::ProcessGet(
        const std::vector<Blob>& kv,
        std::vector<Blob>* result) {
    assert(kv.size() == 1);
    Blob blob = kv[0];
    Op type = GetOpType(blob); 

    if (type == Op::DOTPROD) {
        DotProdParam* param = DotProdParam::FromBlob(blob);
        DotProdResult ret;

        int src_num = param->src.size();
        int dst_num = param->dst.size();
        int K = param->K;
        ret.scale.resize(dst_num);

        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < src_num; ++ i) {
            size_t src_offset = size_t(param->src[i]) * num_cols_local_;
            for (int k = 0; k < K + 1; ++ k) {
                int idx = i * (K + 1) + k;
                size_t dst_offset = size_t(param->dst[idx]) * num_cols_local_;
                real ip = 0;
                for (int j = 0; j < num_cols_local_; ++ j) {
                    ip += W_IN_[src_offset + j] * W_OUT_[dst_offset + j];
                }
                ret.scale[idx] = ip;
            }
        }

        delete param;
        result->push_back(ret.ToBlob());
        multiverso::Log::Debug("[ProcessDotProd] Rank %d (Server %d), #num_edges=%d\n",
            rank_, server_id_, dst_num);
    } else if (type == Op::ADJUST) {
        AdjustParam* param = AdjustParam::FromBlob(blob);
        int src_nodes = param->src.size();
        int dst_nodes = param->dst.size();
        int K = param->K;

        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < src_nodes; ++ i) {
            size_t src_offset = size_t(param->src[i]) * num_cols_local_;
            for (int k = 0; k < K + 1; ++ k) {
                int idx = i * (K + 1) + k; 
                size_t dst_offset = size_t(param->dst[idx]) * num_cols_local_;
                for (int j = 0; j < num_cols_local_; ++ j) {
                    DW_IN_[src_offset + j] += param->scale[idx] * W_OUT_[dst_offset + j];
                    DW_OUT_[dst_offset + j] += param->scale[idx] * W_IN_[src_offset + j];
                }
            }
        }

        std::vector<integer> src_unique = param->src;
        std::vector<integer> dst_unique = param->dst;
        SortEraseDuplicate(src_unique);
        SortEraseDuplicate(dst_unique);

        int num_src_unique = src_unique.size();
        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < num_src_unique; ++ i) {
            size_t offset = src_unique[i] * size_t(num_cols_local_);
            for (int j = 0; j < num_cols_local_; ++ j) {
                W_IN_[offset + j] += DW_IN_[offset + j];
                DW_IN_[offset + j] = 0;
            }
        }

        int num_dst_unique = dst_unique.size();
        #pragma omp parallel for num_threads(num_threads_)
        for (int i = 0; i < num_dst_unique; ++ i) {
            size_t offset = dst_unique[i] * size_t(num_cols_local_);
            for (int j = 0; j < num_cols_local_; ++ j) {
                W_OUT_[offset + j] += DW_OUT_[offset + j];
                DW_OUT_[offset + j] = 0;
            }
        }

        Blob blob(sizeof(int));
        reinterpret_cast<int*>(blob.data())[0] = (int)Op::ADJUST;
        result->push_back(blob);
        delete param;
        multiverso::Log::Debug("[ProcessAdjust] Rank %d (Server %d), #num_edges=%d\n",
            rank_, server_id_, dst_nodes);
    } else if (type == Op::GET) {
        GetParam* param = GetParam::FromBlob(blob); 
        int num_nodes = param->src.size(); 

        GetResult ret;
        ret.server_id = server_id_;
        ret.cols_own = num_cols_local_;
        ret.cols_offset = offset_;
        ret.W.resize(num_nodes * num_cols_local_);

        #pragma omp parallel for num_threads(num_threads_)
        for (size_t i = 0; i < num_nodes; ++ i) {
            size_t dst_offset = i * num_cols_local_;
            size_t src_offset = param->src[i] * num_cols_local_;
            memcpy(ret.W.data() + dst_offset, W_IN_.data() + src_offset, 
                    num_cols_local_ * sizeof(real));
        }

        result->push_back(ret.ToBlob());
        delete param;
        multiverso::Log::Debug("[ProcessGet] Rank %d (Server %d), #num_nodes=%d\n",
            rank_, server_id_, num_nodes);
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
