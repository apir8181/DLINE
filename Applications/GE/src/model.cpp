#include <cassert>
#include <cstdio>
#include <sstream>
#include <string>
#include <pthread.h>
#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include "model.h"
#include "util.h"

namespace graphembedding {

Model::Model(Option* option) : option_(option) {
    communicator_ = NULL;
    graph_partition_ = NULL;
}

Model::~Model() {
    DataBlock* db;

    preload_thread_.join();
    while (preload_queue_.Pop(db)) delete db;

    for (int i = 0; i < preprocess_thread_.size(); ++ i)
        preprocess_thread_[i].join();
    while (preprocess_queue_.Pop(db)) delete db;

    if (communicator_ != NULL) delete communicator_;
    if (graph_partition_ != NULL) delete graph_partition_;
    multiverso::MV_ShutDown();
}

void Model::Init() {
    if (option_->debug) {
        multiverso::Log::ResetLogLevel(multiverso::LogLevel::Debug);
    }

    int argc = 1;
    multiverso::MV_Init(&argc, NULL);
    multiverso::MV_Barrier();
    multiverso::Log::Info("MV Rank %d multiverso initialized\n", 
        multiverso::MV_Rank());

    option_->sample_edges /= multiverso::MV_NumWorkers();

    communicator_ = new (std::nothrow)Communicator(option_);
    assert(communicator_ != NULL);
    multiverso::MV_Barrier();
    multiverso::Log::Info("MV Rank %d prepared table\n", 
        multiverso::MV_Rank());

    if (multiverso::MV_WorkerId() == -1) {
        preload_queue_.Exit();        
        preprocess_queue_.Exit();
    } else {
        graph_partition_ = new (std::nothrow)GraphPartition(option_);
        assert(graph_partition_ != NULL);
        multiverso::Log::Info("MV Rank %d preprocessed graph partition\n", 
            multiverso::MV_Rank());

        preload_thread_ = std::thread(&Model::PreloadThreadMain, this);
        pthread_setname_np(preload_thread_.native_handle(), "preload");
        multiverso::Log::Info("MV Rank %d started preload thread\n", 
            multiverso::MV_Rank());

        preprocess_thread_.clear();
        preprocess_name_.clear();
        preprocess_alive_ = 0;
        for (int i = 0; i < option_->preprocess_threads; ++ i) {
            preprocess_thread_.push_back(std::thread(
                &Model::PreprocessThreadMain, this, i));
            std::string name = "preprocess_" + std::to_string(i);
            preprocess_name_.push_back(name);
            pthread_setname_np(
                preprocess_thread_[i].native_handle(),
                preprocess_name_[i].c_str());
            multiverso::Log::Info("MV Rank %d started preprocess thread %d\n",
                multiverso::MV_Rank(), i);
        }
    }
    multiverso::MV_Barrier();
}

void Model::Train() {
    integerL edge_processed = 0, block_processed = 0;

    multiverso::Log::Info("Rank %d start training\n", multiverso::MV_Rank());
    DataBlock* db = NULL;
    while (preprocess_queue_.Pop(db)) {
        // GET 
        PRINT_CLOCK_BEGIN(get);
        db->AllocParameters();
        communicator_->RequestParameter(db);
        db->MakeParametersCopy();
        PRINT_CLOCK_END(get, "get");
        multiverso::Log::Debug("Rank %d get inputs %d outputs %d\n",
                multiverso::MV_Rank(), 
                db->input_nodes.size(), db->output_nodes.size());

        // train
        PRINT_CLOCK_BEGIN(train);
        real lr = option_->init_learning_rate;
        std::vector<real> loss(option_->compute_threads, 0);
        #pragma omp parallel for num_threads(option_->compute_threads)
        for (int i = 0; i < option_->compute_threads; ++ i) {
            ThreadTrain(db, lr, i, &loss[i]);
        }
        PRINT_CLOCK_END(train, "train");

        // ADD 
        PRINT_CLOCK_BEGIN(add);
        db->MakeParametersDiff();
        communicator_->AddParameter(db);
        PRINT_CLOCK_END(add, "add");

        real total_loss = 0;
        for (auto val : loss) total_loss += val / option_->compute_threads;
        edge_processed += db->Size();
        real progress = std::min((real)1.0,
            (real) edge_processed / option_->sample_edges);
        block_processed ++;
        multiverso::Log::Info("Rank %d iter %lld, loss %f, progress %f\n",
            multiverso::MV_Rank(), block_processed, total_loss, progress);
        
        delete db;
    }
    multiverso::Log::Info("Rank %d train finished\n", multiverso::MV_Rank());
    multiverso::MV_Barrier();
}

void Model::Save() {
    if (multiverso::MV_ServerId() == -1) return;
    
    multiverso::Log::Info("Rank %d saving parameters\n", multiverso::MV_Rank());
    integer row_offset, size;
    size = option_->num_nodes / multiverso::MV_NumServers();
    int server_id = multiverso::MV_ServerId();
    if (size > 0) {
        row_offset = size * server_id;
        if (server_id == multiverso::MV_NumServers() - 1) {
            size = option_->num_nodes - row_offset;
        }
    } else {
        size = server_id < option_->num_nodes ? 1 : 0;
        row_offset = server_id;
    }

    std::stringstream ss;
    ss << option_->output_file << "_part_" << multiverso::MV_Rank();
    std::ofstream fs(ss.str());
    if (!fs.is_open()) {
        multiverso::Log::Fatal("Rank %d can't open output file %s\n",
            multiverso::MV_Rank(), option_->output_file);
    }

    const integer BATCH_SIZE = 100000;
    real* buffer = new (std::nothrow)real[BATCH_SIZE * option_->embedding_size];
    assert(buffer != NULL);
    std::vector<integer> nodes;
    std::vector<real*> vecs;
    for (integer i = 0; i < size; i += BATCH_SIZE) {
        integer remain_size = i + BATCH_SIZE <= size ? BATCH_SIZE : size - i;
        // request local parameter
        nodes.resize(remain_size);
        vecs.resize(remain_size);
        for (integer j = 0; j < remain_size; ++ j) {
            nodes[j] = row_offset + i + j;
            vecs[j] = &buffer[j * option_->embedding_size];
        }
        communicator_->RequestParameterWI(nodes, vecs);
        // save local parameter
        for (integer j = 0; j < remain_size; ++ j) {
            fs << row_offset + i + j;
            real* x = vecs[i + j];
            for (int k = 0; k < option_->embedding_size; ++ k) {
                fs << ' ' << x[k];
            }
            fs << '\n';
        }
    }
    delete buffer;
    fs.close();
}

void Model::PreloadThreadMain() {
    const int PRELOAD_MAX = 3;
    DataBlock* db;
    while (true) {
        PRINT_CLOCK_BEGIN(preload);
        bool success = graph_partition_->ReadDataBlock(db);
        if (!success) break;
        preload_queue_.Push(db);
        PRINT_CLOCK_END(preload, "preload");

        while (preload_queue_.Size() >= PRELOAD_MAX) {
            std::chrono::milliseconds dura(20);
            std::this_thread::sleep_for(dura);
        }
    }
    preload_queue_.Exit();
    multiverso::Log::Info("Rank %d preload thread finished\n", multiverso::MV_Rank());
}

void Model::PreprocessThreadMain(int thread_idx) {
    const int PREPROCESS_MAX = 3;
    preprocess_alive_ ++;

    DataBlock* db;
    while (preload_queue_.Pop(db)) {
        PRINT_CLOCK_BEGIN(preprocess);
        ThreadPreprocess(db, thread_idx);
        preprocess_queue_.Push(db);
        PRINT_CLOCK_END(preprocess, "preprocess");
        while (preprocess_queue_.Size() >= PREPROCESS_MAX) {
            std::chrono::milliseconds dura(20);
            std::this_thread::sleep_for(dura);
        }
    }
    multiverso::Log::Info("Rank %d preprocess thread %d finished\n", 
        multiverso::MV_Rank(), thread_idx);

    preprocess_alive_ --;
    if (preprocess_alive_ == 0) {
        preprocess_queue_.Exit();
    }
}

}
