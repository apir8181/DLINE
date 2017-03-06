#include <fstream>
#include <multiverso/util/log.h>
#include "host_rule.h"

namespace graphembedding {

HostRule::HostRule(const Option* option) : option_(option) {
    const char* hostname_path = "/etc/hostname";
    std::ifstream iFILE(hostname_path);
    if (!iFILE.is_open()) {
        multiverso::Log::Fatal("Unable to open %s\n", hostname_path);
    }
    iFILE >> local_host_name_;
    iFILE.close();

    iFILE.open(option_->rule_file);
    if (!iFILE.is_open()) {
        multiverso::Log::Fatal("Unable to open %s\n", option_->rule_file);
    }
    std::string host, rule;
    while (iFILE >> host >> rule) {
        host2rule_[host] = rule;
    }
}

const char* HostRule::GetLocalHostName() {
    return local_host_name_.c_str();
}

const char* HostRule::GetRule() {
    if (host2rule_.find(local_host_name_) == host2rule_.end()) {
        return "default";
    }
    std::string rule = host2rule_[local_host_name_];
    if (rule == "default" || rule == "server" || rule == "worker") {
        return rule.c_str();
    } else {
        return "default";
    }
}

}
