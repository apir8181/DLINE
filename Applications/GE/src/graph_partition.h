#ifndef GE_PARTITION_H
#define GE_PARTITION_H

#include <cstdio>
#include <string>
#include "constant.h"
#include "util.h"

namespace graphembedding {

struct Edge {
    integer src, dst;
    real weight;
};


class GraphPartition {
public:
    GraphPartition(const Option* o);

    ~GraphPartition();

    void ResetStream();

    std::vector<Edge> ReadDataBlock();

protected:
    const Option* option_;
    std::string file_path_; 
    int rank_, worker_id_;
    FILE* pFILE_;
    integerL edges_in_file_, edges_remained_;
};

}

#endif
