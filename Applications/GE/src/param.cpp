
#include <cassert>
#include "constant.h"
#include "param.h"
#include "util.h"

namespace graphembedding {

Op GetOpType(const multiverso::Blob& blob) {
    return (Op)reinterpret_cast<int*>(blob.data())[0];
}

multiverso::Blob DotProdParam::ToBlob() {
    int src_nodes = src.size();
    int dst_nodes = dst.size();
    int K = (dst_nodes - src_nodes) / src_nodes;
    assert((dst_nodes - src_nodes) % src_nodes == 0);

    size_t blob_size = 3 * sizeof(int) + \
                       src_nodes * sizeof(integer) + \
                       dst_nodes * sizeof(integer);
    multiverso::Blob blob(blob_size);
    char* data = blob.data();

    reinterpret_cast<int*>(data)[0] = (int)Op::DOTPROD;
    data += sizeof(int);

    reinterpret_cast<int*>(data)[0] = src_nodes;
    data += sizeof(int);

    reinterpret_cast<int*>(data)[0] = K;
    data += sizeof(int);

    memcpy(data, src.data(), src_nodes * sizeof(integer));
    data += src_nodes * sizeof(integer);

    memcpy(data, dst.data(), dst_nodes * sizeof(integer));
    data += dst_nodes * sizeof(integer);

    return blob;
}

DotProdParam* DotProdParam::FromBlob(const multiverso::Blob& blob) {
    DotProdParam* param = new DotProdParam();
    char* blob_data = blob.data();

    assert(reinterpret_cast<int*>(blob_data)[0] == (int)Op::DOTPROD);
    blob_data += sizeof(int);

    int src_nodes = reinterpret_cast<int*>(blob_data)[0];
    blob_data += sizeof(int);

    param->K = reinterpret_cast<int*>(blob_data)[0];
    int dst_nodes = (param->K + 1) * src_nodes;
    blob_data += sizeof(int);

    param->src.resize(src_nodes);
    memcpy(param->src.data(), blob_data, src_nodes * sizeof(integer));
    blob_data += src_nodes * sizeof(integer);

    param->dst.resize(dst_nodes);
    memcpy(param->dst.data(), blob_data, dst_nodes * sizeof(integer));
    blob_data += dst_nodes * sizeof(integer);

    return param;
}

multiverso::Blob DotProdResult::ToBlob() {
    int num_pairs = scale.size();
    size_t blob_size = 2 * sizeof(int) + num_pairs * sizeof(real);
    multiverso::Blob blob(blob_size);
    char* blob_data = blob.data();
    
    reinterpret_cast<int*>(blob_data)[0] = (int)Op::DOTPROD;
    blob_data += sizeof(int);

    reinterpret_cast<int*>(blob_data)[0] = num_pairs;
    blob_data += sizeof(int);

    memcpy(blob_data, scale.data(), num_pairs * sizeof(real));
    blob_data += num_pairs * sizeof(real);

    return blob;
}

DotProdResult* DotProdResult::FromBlob(const multiverso::Blob& blob) {
    DotProdResult* result = new DotProdResult();
    char* blob_data = blob.data();

    assert(reinterpret_cast<int*>(blob_data)[0] == (int)Op::DOTPROD);
    blob_data += sizeof(int);

    int num_pairs = reinterpret_cast<int*>(blob_data)[0];
    blob_data += sizeof(int);

    result->scale.resize(num_pairs);
    memcpy(result->scale.data(), blob_data, num_pairs * sizeof(real));
    blob_data += num_pairs * sizeof(real);

    return result;
}

multiverso::Blob AdjustParam::ToBlob() {
    int src_nodes = src.size();
    int dst_nodes = dst.size();
    int K = (dst_nodes - src_nodes) / src_nodes;
    assert((dst_nodes - src_nodes) % src_nodes == 0);
    assert(scale.size() == dst_nodes);
    
    size_t blob_size = 3 * sizeof(int) +\
                       src_nodes * sizeof(integer) +\
                       dst_nodes * sizeof(integer) +\
                       dst_nodes * sizeof(real);
    multiverso::Blob blob(blob_size);
    char* blob_data = blob.data();

    reinterpret_cast<int*>(blob_data)[0] = (int)Op::ADJUST;
    blob_data += sizeof(int);

    reinterpret_cast<int*>(blob_data)[0] = src_nodes;
    blob_data += sizeof(int);

    reinterpret_cast<int*>(blob_data)[0] = K;
    blob_data += sizeof(int);

    memcpy(blob_data, src.data(), src_nodes * sizeof(integer));
    blob_data += src_nodes * sizeof(integer);

    memcpy(blob_data, dst.data(), dst_nodes * sizeof(integer));
    blob_data += dst_nodes * sizeof(integer);

    memcpy(blob_data, scale.data(), dst_nodes * sizeof(real));
    blob_data += dst_nodes * sizeof(real);

    return blob;
}

AdjustParam* AdjustParam::FromBlob(const multiverso::Blob& blob) {
    AdjustParam* param = new AdjustParam();
    char* blob_data = blob.data();

    assert(reinterpret_cast<int*>(blob_data)[0] == (int)Op::ADJUST);
    blob_data += sizeof(int);

    int src_nodes = reinterpret_cast<int*>(blob_data)[0];
    blob_data += sizeof(int);

    param->K = reinterpret_cast<int*>(blob_data)[0];
    blob_data += sizeof(int);

    param->src.resize(src_nodes);
    memcpy(param->src.data(), blob_data, src_nodes * sizeof(integer));
    blob_data += src_nodes * sizeof(integer);

    int dst_nodes = (param->K + 1) * src_nodes;
    param->dst.resize(dst_nodes);
    memcpy(param->dst.data(), blob_data, dst_nodes * sizeof(integer));
    blob_data += dst_nodes * sizeof(integer);

    param->scale.resize(dst_nodes);
    memcpy(param->scale.data(), blob_data, dst_nodes * sizeof(real));
    blob_data += dst_nodes * sizeof(real);

    return param;
}

multiverso::Blob GetParam::ToBlob() {
    int num_nodes = src.size();
    
    size_t blob_size = 2 * sizeof(int) + num_nodes * sizeof(integer);
    multiverso::Blob blob(blob_size);
    char* blob_data = blob.data();

    reinterpret_cast<int*>(blob_data)[0] = (int)Op::GET;
    blob_data += sizeof(int);

    reinterpret_cast<int*>(blob_data)[0] = num_nodes;
    blob_data += sizeof(int);

    memcpy(blob_data, src.data(), num_nodes * sizeof(integer));
    blob_data += num_nodes * sizeof(integer);

    return blob;
}

GetParam* GetParam::FromBlob(const multiverso::Blob& blob) {
    GetParam* param = new GetParam();
    char* blob_data = blob.data();
    
    assert(reinterpret_cast<int*>(blob_data)[0] == (int)Op::GET);
    blob_data += sizeof(int);

    int num_nodes = reinterpret_cast<int*>(blob_data)[0];
    blob_data += sizeof(int);

    param->src.resize(num_nodes);
    memcpy(param->src.data(), blob_data, num_nodes * sizeof(integer));
    blob_data += num_nodes * sizeof(integer);

    return param;
}

multiverso::Blob GetResult::ToBlob() {
    int num_elems = W.size();
    size_t blob_size = 5 * sizeof(int) + num_elems * sizeof(real);
    
    multiverso::Blob blob(blob_size);
    char* blob_data = blob.data();

    reinterpret_cast<int*>(blob_data)[0] = (int)Op::GET;
    blob_data += sizeof(int);

    reinterpret_cast<int*>(blob_data)[0] = num_elems;
    blob_data += sizeof(int);

    reinterpret_cast<int*>(blob_data)[0] = server_id;
    blob_data += sizeof(int);

    reinterpret_cast<int*>(blob_data)[0] = cols_own;
    blob_data += sizeof(int);

    reinterpret_cast<int*>(blob_data)[0] = cols_offset;
    blob_data += sizeof(int);

    memcpy(blob_data, W.data(), num_elems * sizeof(real));
    blob_data += num_elems * sizeof(real);

    return blob;
}

GetResult* GetResult::FromBlob(const multiverso::Blob& blob) {
    GetResult* result = new GetResult();
    char* blob_data = blob.data();
    
    assert(reinterpret_cast<int*>(blob_data)[0] == (int)Op::GET);
    blob_data += sizeof(int);

    int num_elems = reinterpret_cast<int*>(blob_data)[0];
    blob_data += sizeof(int);

    result->server_id = reinterpret_cast<int*>(blob_data)[0];
    blob_data += sizeof(int);

    result->cols_own = reinterpret_cast<int*>(blob_data)[0];
    blob_data += sizeof(int);

    result->cols_offset = reinterpret_cast<int*>(blob_data)[0];
    blob_data += sizeof(int);

    result->W.resize(num_elems);
    memcpy(result->W.data(), blob_data, num_elems * sizeof(real));
    blob_data += num_elems * sizeof(real);

    return result;
}

}
