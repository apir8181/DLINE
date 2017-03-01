#ifndef GE_MEM_SAVE_MATRIX_TABLE_H
#define GE_MEM_SAVE_MATRIX_TABLE_H

#include <mutex>
#include <multiverso/table_interface.h>
#include "constant.h"

namespace graphembedding {

template<typename T>
struct MemSaveMatrixTableOption;

template<typename T>
class MemSaveMatrixWorkerTable: public multiverso::WorkerTable {
protected:
    using Blob = multiverso::Blob;

public:
    MemSaveMatrixWorkerTable(const MemSaveMatrixTableOption<T> &option);

    ~MemSaveMatrixWorkerTable();

    void Get(const std::vector<integer>& keys, const std::vector<T*> &vecs);

    void Add(const std::vector<integer>& keys, const std::vector<T*> &vecs);

    int Partition(const std::vector<Blob>& kv, multiverso::MsgType,
        std::unordered_map<int, std::vector<Blob> >* out);

    void ProcessReplyGet(std::vector<Blob>& reply_data);

private:
    int get_reply_count_;
    integer num_rows_;
    int num_cols_;
    int num_servers_;
    size_t row_size_;
    T** data_;
};

template<typename T>
class MemSaveMatrixServerTable: public multiverso::ServerTable {
protected:
    using Blob = multiverso::Blob;

public:
    MemSaveMatrixServerTable(const MemSaveMatrixTableOption<T>& option);
    
    ~MemSaveMatrixServerTable();
    
    void ProcessAdd(const std::vector<Blob>& data);

    void ProcessGet(const std::vector<Blob>& data, std::vector<Blob>* result);

    void Store(multiverso::Stream* s);

    void Load(multiverso::Stream *s);

private:
    integer num_rows_, num_rows_local_;
    int num_cols_;
    int num_servers_, server_id_;
    size_t offset_, row_size_;
    std::vector<T> storage_;
};

template<typename T>
struct MemSaveMatrixTableOption {
    integer num_rows;
    int num_cols;
    T min_val, max_val;
    MemSaveMatrixTableOption(integer r, int c, T min_v, T max_v)
        : num_rows(r), num_cols(c), min_val(min_v), max_val(max_v) {}
    DEFINE_TABLE_TYPE(T, MemSaveMatrixWorkerTable, MemSaveMatrixServerTable);
};

}

#endif
