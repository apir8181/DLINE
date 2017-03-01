#ifndef GE_DICTIONARY_H
#define GE_DICTIONARY_H

#include <random>
#include <vector>
#include "constant.h"
#include "util.h"

namespace graphembedding {

class Dictionary {
private:
    void Initialization();
    const Option* option_;
    std::vector<integer> node_id_;
    AliasMethod* alias_method_;

public:
    Dictionary(const Option* option);

    ~Dictionary();

    inline integer Sample(std::mt19937_64 &gen) {
        integer idx = alias_method_->Sample(gen);
        return node_id_[idx];
    }
};

}

#endif
