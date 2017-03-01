#include <random>
#include <cassert>
#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include <multiverso/updater/updater.h>
#include <multiverso/io/io.h>
#include "mem_save_matrix_table.h"
#include "util.h"

namespace graphembedding {

template<typename T>
MemSaveMatrixWorkerTable<T>::MemSaveMatrixWorkerTable(
        const MemSaveMatrixTableOption<T>& option) : WorkerTable() {
    get_reply_count_ = 0;
    num_rows_ = option.num_rows;
    num_cols_ = option.num_cols;
    num_servers_ = multiverso::MV_NumServers();
    row_size_ = num_cols_ * sizeof(T);
    data_ = new (std::nothrow)T*[num_rows_]();
    assert(data_ != NULL);
    assert(num_rows_ >= num_servers_); 
}

template<typename T>
MemSaveMatrixWorkerTable<T>::~MemSaveMatrixWorkerTable() {
    if (data_ != NULL) delete[] data_;
}

template<typename T>
void MemSaveMatrixWorkerTable<T>::Get(
        const std::vector<integer>& row_ids,
        const std::vector<T*> &data_vecs) {
    assert(row_ids.size() == data_vecs.size());
    assert(get_reply_count_ == 0);
    size_t keys_size = row_ids.size();
    for (auto i = 0; i < keys_size; ++ i) {
        data_[row_ids[i]] = data_vecs[i];
    }
    Blob keys(row_ids.data(), keys_size * sizeof(integer)); 
    WorkerTable::Get(keys, NULL);
    multiverso::Log::Debug("[Get] worker = %d, #rows_set = %lld\n",
        multiverso::MV_Rank(), keys_size);
}

template<typename T>
void MemSaveMatrixWorkerTable<T>::Add(
        const std::vector<integer>& row_ids,
        const std::vector<T*>& data_vecs) {
    Blob ids_blob(row_ids.data(), row_ids.size() * sizeof(integer));
    Blob data_blob(row_ids.size() * row_size_);
    assert(get_reply_count_ == 0);
    T* dst_vecs = reinterpret_cast<T*>(data_blob.data());
    size_t keys_size = row_ids.size();
    for (auto i = 0; i < keys_size; ++ i) {
        memcpy(dst_vecs + i * num_cols_, data_vecs[i], row_size_);
    }
    WorkerTable::Add(ids_blob, data_blob, NULL); 
    multiverso::Log::Debug("[Add] worker = %d, #row = %lld\n",
        multiverso::MV_Rank(), keys_size); 
}

template<typename T>
int MemSaveMatrixWorkerTable<T>::Partition(
        const std::vector<Blob>& kv, multiverso::MsgType,
        std::unordered_map<int, std::vector<Blob> >* out) {
    integer* keys = reinterpret_cast<integer*>(kv[0].data());
    size_t keys_size = kv[0].size<integer>();

    //count row number in each server
    std::vector<int> dest;
    std::vector<size_t> count;
    count.resize(num_servers_, 0);
    integer num_row_each = num_rows_ / num_servers_;
    for (auto i = 0; i < keys_size; ++i) {
        int dst = keys[i] / num_row_each;
        dst = (dst >= num_servers_ ? num_servers_ - 1 : dst);
        dest.push_back(dst);
        ++ count[dst];
    }

    // allocate memory for blobs
    for (auto i = 0; i < num_servers_; i++) {     
        int rank = multiverso::MV_ServerIdToRank(i);
        if (count[i] != 0) {
            std::vector<Blob>& vec = (*out)[rank];
            vec.push_back(Blob(count[i] * sizeof(integer)));
            if (kv.size() >= 2) vec.push_back(Blob(count[i] * row_size_));
        } 
        /*
        else {
            std::vector<Blob>& vec = (*out)[rank];
            vec.push_back(Blob(sizeof(integer)));
            if (kv.size() >= 2) vec.push_back(Blob(sizeof(integer)));
            reinterpret_cast<integer*>(vec[0].data())[0] = -1;
        }
        */
    }
    count.clear();
    count.resize(num_servers_, 0);

    T* src_vec = kv.size() == 2 ? reinterpret_cast<T*>(kv[1].data()) : NULL;
    for (auto i = 0; i < keys_size; ++i) {
        int dst = dest[i];
        int rank = multiverso::MV_ServerIdToRank(dst);
        (*out)[rank][0].As<integer>(count[dst]) = keys[i];
        if (kv.size() == 2) { // copy values
            T* dst_vec = reinterpret_cast<T*>((*out)[rank][1].data());
            memcpy(dst_vec + count[dst] * num_cols_, src_vec + i * num_cols_, row_size_);
        }
        ++count[dst];
    }

    assert(get_reply_count_ == 0);
    if (kv.size() == 1) get_reply_count_ = out->size();

    return static_cast<int>(out->size());
}

template <typename T>
void MemSaveMatrixWorkerTable<T>::ProcessReplyGet(std::vector<Blob>& reply_data) {
    assert(reply_data.size() == 2);
    get_reply_count_ --;

    size_t keys_size = reply_data[0].size<integer>();
    integer* keys = reinterpret_cast<integer*>(reply_data[0].data());
    T* values = reinterpret_cast<T*>(reply_data[1].data());
    //if (keys_size == 1 && keys[0] == -1) return;
    for (auto i = 0; i < keys_size; ++i) { 
        memcpy(data_[keys[i]], values + i * num_cols_, row_size_);
    }
    multiverso::Log::Debug("[ProcessReplyGet] worker = %d, #rows_set = %lld\n",
        multiverso::MV_Rank(), keys_size);
}

template <typename T>
MemSaveMatrixServerTable<T>::MemSaveMatrixServerTable(
        const MemSaveMatrixTableOption<T>& option) : ServerTable() {
    num_rows_ = option.num_rows;
    num_cols_ = option.num_cols;
    num_servers_ = multiverso::MV_NumServers();
    server_id_ = multiverso::MV_ServerId();
    assert(num_rows_ >= num_servers_);

    row_size_ = num_cols_ * sizeof(T);
    num_rows_local_ = num_rows_ / num_servers_;
    offset_ = num_rows_local_ * multiverso::MV_ServerId();
    if (server_id_ == num_servers_ - 1) {
        num_rows_local_ = num_rows_ - offset_;
    }

    // create storage data
    size_t total_size = size_t(num_rows_local_) * num_cols_;
    storage_.resize(total_size);

    // initialization 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(option.min_val, option.max_val);
    size_t storage_size = storage_.size();
    for (auto i = 0; i < storage_size; i ++) {
      storage_[i] = dis(gen);
    }

    multiverso::Log::Info("[Init] Server = %d, type = memSaveMatrixTable, size = [%lld x %lld], "
        "local size = [%lld x %lld]\n",
        server_id_, num_rows_, num_cols_, num_rows_local_, num_cols_);
}

template<typename T>
MemSaveMatrixServerTable<T>::~MemSaveMatrixServerTable() {
    storage_.clear();
}

template<typename T>
void MemSaveMatrixServerTable<T>::ProcessAdd(const std::vector<Blob>& data) {
    assert(data.size() == 2);
    size_t keys_size = data[0].size<integer>();
    integer* keys = reinterpret_cast<integer*>(data[0].data());
    T* vals = reinterpret_cast<T*>(data[1].data());
    T* sdata = storage_.data();
    //if (keys_size == 1 && keys[0] == -1) return;
    for (auto i = 0; i < keys_size; ++ i) {
        size_t offset_s = (keys[i] - offset_) * num_cols_;
        size_t offset_v = i * num_cols_;
        for (auto j = 0; j < num_cols_; ++ j)
            sdata[offset_s + j] += vals[offset_v + j];
    }
    multiverso::Log::Debug("[ProcessAdd] Server = %d, adding #rows = %lld\n",
        server_id_, keys_size);
}

template<typename T>
void MemSaveMatrixServerTable<T>::ProcessGet(const std::vector<Blob>& data,
        std::vector<Blob>* result) {
    assert(data.size() == 1);
    size_t keys_size = data[0].size<integer>(); 
    integer* keys = reinterpret_cast<integer*>(data[0].data());

    result->push_back(data[0]);
    result->push_back(Blob(keys_size * row_size_));

    //if (keys_size == 1 && keys[0] == -1) return;

    T* sdata = storage_.data();
    T* vals = reinterpret_cast<T*>((*result)[1].data());
    for (auto i = 0; i < keys_size; ++ i) {
        memcpy(vals + i * num_cols_, sdata + (keys[i] - offset_) * num_cols_, row_size_);
    }
    multiverso::Log::Debug("[ProcessGet] Server = %d, getting row #rows = %d\n",
        server_id_, keys_size);
}

template<typename T>
void MemSaveMatrixServerTable<T>::Store(multiverso::Stream* s) {
    // TODO
}

template<typename T>
void MemSaveMatrixServerTable<T>::Load(multiverso::Stream *s) {
    // TODO
}

MV_INSTANTIATE_CLASS_WITH_REAL_TYPE(MemSaveMatrixWorkerTable);
MV_INSTANTIATE_CLASS_WITH_REAL_TYPE(MemSaveMatrixServerTable);

}
