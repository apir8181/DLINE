#ifndef GE_UTIL_H 
#define GE_UTIL_H 

#include <random>
#include <vector>
#include <time.h>
#include <multiverso/multiverso.h>
#include <multiverso/util/log.h>
#include "constant.h"

namespace graphembedding {

enum class AlgorithmType {
    GF,
    Line
};

struct Option {
    const char* graph_part_file;
    const char* output_file;
    // common arguments 
    AlgorithmType algo_type; 
    bool debug;
    int embedding_size, compute_threads, preprocess_threads;
    integer num_nodes;
    integerL sample_edges, block_num_edges;
    real init_learning_rate;
    // LINE arguments
    const char* dict_file;
    int negative_num;

    Option();
    void ParseArgs(int argc, char* argv[]);
    void PrintArgs();
    void PrintUsage();
};


class AliasMethod {
private:
    size_t N_;
    std::vector<real> prob_;
    std::vector<integer> shared_;
    std::uniform_real_distribution<real> dist_;

public:
    AliasMethod(const std::vector<real>& weight);
    
    ~AliasMethod();

    inline integer Sample(std::mt19937_64& gen) {
        real rnd_value_1 = dist_(gen), rnd_value_2 = dist_(gen);
        integer idx = (integer)(rnd_value_1 * N_);
        idx = idx < N_ ? idx : N_ - 1;
        return prob_[idx] < rnd_value_2 ? idx : shared_[idx]; 
    };
};

template<typename T>
integerL BinarySearch(const std::vector<T>& array, T target);

template<typename T>
void SortEraseDuplicate(std::vector<T>& array);

#define PRINT_CLOCK_BEGIN(name)                         \
    struct timespec start_##name;                       \
    clock_gettime(CLOCK_MONOTONIC, &start_##name); 

#define PRINT_CLOCK_END(name, msg)                      \
    struct timespec end_##name;                         \
    clock_gettime(CLOCK_MONOTONIC, &end_##name);        \
    real time_##name = end_##name.tv_sec -              \
            start_##name.tv_sec;                        \
    time_##name += (end_##name.tv_nsec -                \
            start_##name.tv_nsec) / 1e9f;               \
    multiverso::Log::Debug("Rank %d run %s time %lf\n", \
        multiverso::MV_Rank(), msg, time_##name);

}

#endif
