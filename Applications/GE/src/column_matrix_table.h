#ifndef GE_COLUMN_MATRIX_TABLE_H
#define GE_COLUMN_MATRIX_TABLE_H

#include <mutex>
#include <vector>
#include <multiverso/table_interface.h>
#include <unordered_set>
#include "constant.h"
#include "param.h"

namespace graphembedding {

template<typename T>
struct ColumnMatrixTableOption;

template<typename T>
class ColumnMatrixWorkerTable: public multiverso::WorkerTable {
protected:
    using Blob = multiverso::Blob;

public:
    ColumnMatrixWorkerTable(const ColumnMatrixTableOption<T> &option);

    ~ColumnMatrixWorkerTable();

    DotProdResult* DotProd(DotProdParam* param);

    void Adjust(AdjustParam* param);

    GetResult* Get(GetParam* param);

    int Partition(const std::vector<Blob>& kv, multiverso::MsgType,
        std::unordered_map<int, std::vector<Blob> >* out);

    void ProcessReplyGet(std::vector<Blob>& reply_data);

private:
    std::mutex mutex_;
    int num_servers_, rank_, worker_id_;
    int num_cols_;
    DotProdResult* dotprod_result_;
    GetResult* get_result_;
};

template<typename T>
class ColumnMatrixServerTable: public multiverso::ServerTable {
protected:
    using Blob = multiverso::Blob;

public:
    ColumnMatrixServerTable(const ColumnMatrixTableOption<T>& option);
    
    ~ColumnMatrixServerTable();

    void ProcessGet(const std::vector<Blob>& data, std::vector<Blob>* result);

    void ProcessAdd(const std::vector<Blob>& kv);

    void Store(multiverso::Stream* s);

    void Load(multiverso::Stream *s);

private:
    integer num_rows_;
    int num_cols_, num_cols_local_, offset_;
    int num_servers_, rank_, server_id_, num_threads_;
    std::vector<T> W_IN_, W_OUT_;
    std::vector<T> DW_IN_, DW_OUT_;
};

template<typename T>
struct ColumnMatrixTableOption {
    integer num_rows;
    int num_cols;
    T min_val, max_val;
    int threads;
    ColumnMatrixTableOption(integer r, int c, T min_v, T max_v, int t)
        : num_rows(r), num_cols(c), min_val(min_v), max_val(max_v), threads(t) {}
    DEFINE_TABLE_TYPE(T, ColumnMatrixWorkerTable, ColumnMatrixServerTable);
};

}

#endif
