#include <cassert>
#include <iostream>
#include <multiverso/util/log.h>
#include "model.h"
#include "util.h"
#include "line_model.h"

using namespace graphembedding;

int main(int argc, char* argv[]) {
    Option* option = new Option();
    option->PrintUsage();
    option->ParseArgs(argc, argv);
    option->PrintArgs();
   
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    Model* model = NULL;
    if (option->algo_type == AlgorithmType::Line) {
        model = new (std::nothrow)LineModel(option);
        assert(model != NULL);
    }
    model->Init();
    model->Train();
    model->Save();
    delete model;

    clock_gettime(CLOCK_MONOTONIC, &finish);
    multiverso::Log::Info("Rank %d train time %ds\n",
        multiverso::MV_Rank(), finish.tv_sec - start.tv_sec); 

    return 0;
}
