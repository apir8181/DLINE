#include <cassert>
#include <cstdio>
#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include "dictionary.h"

namespace graphembedding {

Dictionary::Dictionary(const Option* option) : option_(option) {
    node_id_.clear();
    alias_method_ = NULL;
    Initialization();
}

Dictionary::~Dictionary() {
    if (alias_method_) delete alias_method_;
}

void Dictionary::Initialization() {
    FILE* pFILE = fopen(option_->dict_file, "r");
    if (pFILE == NULL) {
        multiverso::Log::Fatal("Rank %d can't open file %s\n",
            multiverso::MV_Rank(), option_->dict_file);
    } 
    multiverso::Log::Info("Rank %d reading dictionary %s\n",
        multiverso::MV_Rank(), option_->dict_file);

    std::vector<real> node_degree;
    integer id;
    real weight;
    while (fscanf(pFILE, "%d %f", &id, &weight) != EOF) {
        node_id_.push_back(id);
        node_degree.push_back(weight);
    }
    fclose(pFILE);

    if (node_degree.size() == 0) {
        multiverso::Log::Fatal("Rank %d dictionary file %s empty\n",
            multiverso::MV_Rank(), option_->dict_file);
    }

    alias_method_ = new (std::nothrow)AliasMethod(node_degree);
    assert(alias_method_ != NULL);
}

}
