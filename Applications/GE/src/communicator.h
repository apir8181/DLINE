#ifndef GE_COMMUNICATOR_H
#define GE_COMMUNICATOR_H

#include <vector>
#include <multiverso/table/array_table.h>
#include <multiverso/table/matrix_table.h>
#include "mem_save_matrix_table.h"
#include "constant.h"
#include "data_block.h"

namespace graphembedding {

class Communicator {
public:
    Communicator(const Option* o);
    ~Communicator();

    void RequestParameter(DataBlock* datablock);
    void AddParameter(DataBlock* datablock);
    void RequestParameterWI(const std::vector<integer>& nodes, const std::vector<real*>& vecs);

private:
    void PrepareTable();
    void ClearTable();

    const Option* option_;
    MemSaveMatrixWorkerTable<real>* W_IE_table_;
    MemSaveMatrixWorkerTable<real>* W_OE_table_;
};

}
#endif
