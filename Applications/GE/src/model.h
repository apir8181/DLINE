#ifndef GE_MODEL_H
#define GE_MODEL_H

#include <multiverso/util/mt_queue.h>
#include <thread>
#include <vector>
#include <atomic>
#include "constant.h"
#include "data_block.h"
#include "graph_partition.h"
#include "communicator.h"
#include "util.h"

namespace graphembedding {

class Model {
public:
    Model(Option* option);

    virtual ~Model();

    void Init();

    void Train();

    void Save();

protected:
    virtual void ThreadPreprocess(DataBlock* db, int thread_idx) = 0;
    virtual void ThreadTrain(DataBlock* db, real lr, int thread_idx, void* args) = 0;

    Option* option_;

private:
    void PreloadThreadMain();
    void PreprocessThreadMain(int idx);

    Communicator* communicator_;
    GraphPartition* graph_partition_;

    multiverso::MtQueue<DataBlock*> preload_queue_;
    std::thread preload_thread_;

    multiverso::MtQueue<DataBlock*> preprocess_queue_;
    std::vector<std::string> preprocess_name_;
    std::vector<std::thread> preprocess_thread_;
    std::atomic<int> preprocess_alive_;
};

}

#endif
