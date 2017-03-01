#ifndef GE_PARTITION_H
#define GE_PARTITION_H

#include <vector>
#include <cstdio>
#include <string>
#include "constant.h"
#include "data_block.h"
#include "util.h"

namespace graphembedding {

class GraphPartition {
public:
    GraphPartition(const Option* o);

    ~GraphPartition();

    realL TotalWeight();

    void ResetStream();

    bool ReadDataBlock(DataBlock*& db);

protected:
    const Option* option_;
    std::string file_path_; 
    FILE* pFILE_;
    integerL edges_in_file_, edges_remained_;
    realL total_weight_;
};

}

#endif
