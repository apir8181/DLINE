#ifndef GE_HOST_RULE_H
#define GE_HOST_RULE_H

#include <string>
#include <map>
#include "util.h"

namespace graphembedding {

class HostRule {
public:
    HostRule(const Option* option);

    const char* GetLocalHostName();

    const char* GetRule();

private:
    const Option* option_;
    std::string local_host_name_;
    std::map<std::string, std::string> host2rule_;
};

}

#endif
