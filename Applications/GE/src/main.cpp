#include <cassert>
#include <iostream>
#include <multiverso/util/log.h>
#include "model.h"
#include "util.h"

using namespace graphembedding;

int main(int argc, char* argv[]) {
    Option* option = new Option();
    option->PrintUsage();
    option->ParseArgs(argc, argv);
    option->PrintArgs();
   
    Model* model = new Model(option);
    model->Init();

    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);
    model->Train();
    clock_gettime(CLOCK_MONOTONIC, &finish);
    multiverso::Log::Info("Rank %d train time %ds\n",
        multiverso::MV_Rank(), finish.tv_sec - start.tv_sec); 

    model->Save();
    delete model;

    return 0;
}
