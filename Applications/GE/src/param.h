#ifndef GE_PARAM_H
#define GE_PARAM_H

#include <multiverso/multiverso.h>
#include <multiverso/blob.h>

namespace graphembedding {

enum class Op { GET, DOTPROD, ADJUST };

Op GetOpType(const multiverso::Blob& blob);

// Format:
// - src: src1, src2, ..., srcN
// - dst: dst1, dst1_neg1, ..., dst1_negK, ..., dstN_negK 
struct DotProdParam {
    int K;
    std::vector<integer> src;
    std::vector<integer> dst;
    
    multiverso::Blob ToBlob();
    static DotProdParam* FromBlob(const multiverso::Blob& blob);
};

// Format:
// - scale: dst1_IP, dst1_neg1_IP, ..., dst1_negK_IP, ..., dstN_negK_IP 
struct DotProdResult {
    std::vector<real> scale;
    
    multiverso::Blob ToBlob();
    static DotProdResult* FromBlob(const multiverso::Blob& blob);
};

// Format:
// - scale: src1_dst1_err, .., srcN_dstN_err, src1_neg1_err, ..., srcN_negK_err
struct AdjustParam {
    int K;
    std::vector<integer> src;
    std::vector<integer> dst;
    std::vector<real> scale;

    multiverso::Blob ToBlob();
    static AdjustParam* FromBlob(const multiverso::Blob& blob);
};

struct GetParam {
    std::vector<integer> src;

    multiverso::Blob ToBlob();
    static GetParam* FromBlob(const multiverso::Blob& blob);
};

struct GetResult {
    int server_id, cols_own, cols_offset;
    std::vector<real> W;

    multiverso::Blob ToBlob();
    static GetResult* FromBlob(const multiverso::Blob& blob);
};

}

#endif
