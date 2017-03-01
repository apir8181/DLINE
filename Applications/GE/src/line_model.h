#ifndef GE_LINE_MODEL_H
#define GE_LINE_MODEL_H

#include <random>
#include <vector>
#include "model.h"
#include "data_block.h"
#include "dictionary.h"

namespace graphembedding {

class LineModel : public Model {
public:
    LineModel(Option* option);

    virtual ~LineModel();

protected:
    virtual void ThreadPreprocess(DataBlock* db, int thread_idx);
    virtual void ThreadTrain(DataBlock* db, real lr, int thread_idx, void* args);

private:
    real TrainEdge(real* input_vec, real* output_vec, 
            real* input_err, real* output_err, bool label);
    void Update(real* vec, real* err, real lr);
    Dictionary* dictionary_;
};

}

#endif
