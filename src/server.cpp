#include "multiverso/server.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "multiverso/actor.h"
#include "multiverso/dashboard.h"
#include "multiverso/multiverso.h"
#include "multiverso/table_interface.h"
#include "multiverso/io/io.h"
#include "multiverso/util/configure.h"
#include "multiverso/util/mt_queue.h"
#include "multiverso/zoo.h"


namespace multiverso {

MV_DEFINE_bool(sync, false, "sync or async");
MV_DEFINE_int(basync_clock, 3, "max tolerance clock for bounded async server");
MV_DEFINE_int(backup_worker_ratio, 0, "ratio% of backup workers, set 20 means 20%");

Server::Server() : Actor(actor::kServer) {
  RegisterHandler(MsgType::Request_Get, std::bind(
    &Server::ProcessGet, this, std::placeholders::_1));
  RegisterHandler(MsgType::Request_Add, std::bind(
    &Server::ProcessAdd, this, std::placeholders::_1));
}

int Server::RegisterTable(ServerTable* server_table) {
  int id = static_cast<int>(store_.size());
  store_.push_back(server_table);
  return id;
}

void Server::ProcessGet(MessagePtr& msg) {
  MONITOR_BEGIN(SERVER_PROCESS_GET);
  if (msg->data().size() != 0) {
    MessagePtr reply(msg->CreateReplyMessage());
    int table_id = msg->table_id();
    CHECK(table_id >= 0 && table_id < static_cast<int>(store_.size()));
    store_[table_id]->ProcessGet(msg->data(), &reply->data());
    SendTo(actor::kCommunicator, reply);
  }
  MONITOR_END(SERVER_PROCESS_GET);
}

void Server::ProcessAdd(MessagePtr& msg) {
  MONITOR_BEGIN(SERVER_PROCESS_ADD)
  if (msg->data().size() != 0) {
    MessagePtr reply(msg->CreateReplyMessage());
    int table_id = msg->table_id();
    CHECK(table_id >= 0 && table_id < static_cast<int>(store_.size()));
    store_[table_id]->ProcessAdd(msg->data());
    SendTo(actor::kCommunicator, reply);
  }
  MONITOR_END(SERVER_PROCESS_ADD)
}


// The Sync Server implement logic to support Sync SGD training
// The implementation assumes all the workers will call same number
// of Add and/or Get requests
// The server promise all workers i-th Get will get the same parameters
// If worker k has add delta to server j times when its i-th Get 
// then the server will return the parameter after all K 
// workers finished their j-th update
class SyncServer : public Server {
public:
  SyncServer() : Server() {
    RegisterHandler(MsgType::Server_Finish_Train, std::bind(
      &SyncServer::ProcessFinishTrain, this, std::placeholders::_1));
    int num_worker = Zoo::Get()->num_workers();
    worker_get_clocks_.reset(new VectorClock(num_worker));
    worker_add_clocks_.reset(new VectorClock(num_worker));
    num_waited_add_.resize(num_worker, 0);
  }

  // make some modification to suit to the sync server
  // please not use in other place, may different with the general vector clock
  class VectorClock {
  public:
    explicit VectorClock(int n) : 
      local_clock_(n, 0), global_clock_(0), size_(0) {}

    // Return true when all clock reach a same number
    virtual bool Update(int i) {
      ++local_clock_[i];
      if (global_clock_ < *(std::min_element(std::begin(local_clock_),
        std::end(local_clock_)))) {
        ++global_clock_;
        if (global_clock_ == max_element()) {
          return true;
        }
      }
      return false;
    }

    virtual bool FinishTrain(int i) {
      local_clock_[i] = std::numeric_limits<int>::max();
      if (global_clock_ < min_element()) {
        global_clock_ = min_element();
        if (global_clock_ == max_element()) {
          return true;
        }
      }
      return false;
    }

    std::string DebugString() {
      std::string rank = "rank :" + std::to_string(MV_Rank());
      std::string os = rank + " global ";
      os += std::to_string(global_clock_) + " local: ";
      for (auto i : local_clock_) { 
        if (i == std::numeric_limits<int>::max()) os += "-1 ";
        else os += std::to_string(i) + " ";
      }
      return os;
    }

    int local_clock(int i) const { return local_clock_[i]; }
    int global_clock() const { return global_clock_; }

  private:
    int max_element() const {
      int max = global_clock_;
      for (auto val : local_clock_) {
        max = (val != std::numeric_limits<int>::max() && val > max) ? val : max;
      }
      return max;
    }
    int min_element() const {
      return *std::min_element(std::begin(local_clock_), std::end(local_clock_));
    }
  protected:
    std::vector<int> local_clock_;
    int global_clock_;
    int size_;
  };
protected:
  void ProcessAdd(MessagePtr& msg) override {
    // 1. Before add: cache faster worker
    int worker = Zoo::Get()->rank_to_worker_id(msg->src());
    if (worker_get_clocks_->local_clock(worker) >
        worker_get_clocks_->global_clock()) {
      msg_add_cache_.Push(msg);
      ++num_waited_add_[worker];
      return;
    }
    // 2. Process Add
    Server::ProcessAdd(msg);
    // 3. After add: process cached process get if necessary
    if (worker_add_clocks_->Update(worker)) {
      CHECK(msg_add_cache_.Empty());
      while (!msg_get_cache_.Empty()) {
        MessagePtr get_msg;
        CHECK(msg_get_cache_.TryPop(get_msg));
        int get_worker = Zoo::Get()->rank_to_worker_id(get_msg->src());
        Server::ProcessGet(get_msg);
        CHECK(!worker_get_clocks_->Update(get_worker));
      }
    }
  }

  void ProcessGet(MessagePtr& msg) override {
    // 1. Before get: cache faster worker
    int worker = Zoo::Get()->rank_to_worker_id(msg->src());
    if (worker_add_clocks_->local_clock(worker) >
        worker_add_clocks_->global_clock() || 
        num_waited_add_[worker] > 0) {
      // Will wait for other worker finished Add
      msg_get_cache_.Push(msg);
      return;
    }
    // 2. Process Get
    Server::ProcessGet(msg);
    // 3. After get: process cached process add if necessary
    if (worker_get_clocks_->Update(worker)) {
      while (!msg_add_cache_.Empty()) {
        MessagePtr add_msg;
        CHECK(msg_add_cache_.TryPop(add_msg));
        int add_worker = Zoo::Get()->rank_to_worker_id(add_msg->src());
        Server::ProcessAdd(add_msg);
        CHECK(!worker_add_clocks_->Update(add_worker));
        --num_waited_add_[add_worker];
      }
    }
  }

  void ProcessFinishTrain(MessagePtr& msg) {
    int worker = Zoo::Get()->rank_to_worker_id(msg->src());
    if (worker_add_clocks_->FinishTrain(worker)) {
      CHECK(msg_add_cache_.Empty());
      while (!msg_get_cache_.Empty()) {
        MessagePtr get_msg;
        CHECK(msg_get_cache_.TryPop(get_msg));
        int get_worker = Zoo::Get()->rank_to_worker_id(get_msg->src());
        Server::ProcessGet(get_msg);
        CHECK(!worker_get_clocks_->Update(get_worker));
      }
    }
    if (worker_get_clocks_->FinishTrain(worker)) {
      CHECK(msg_get_cache_.Empty());
      while (!msg_add_cache_.Empty()) {
        MessagePtr add_msg;
        CHECK(msg_add_cache_.TryPop(add_msg));
        int add_worker = Zoo::Get()->rank_to_worker_id(add_msg->src());
        Server::ProcessAdd(add_msg);
        CHECK(!worker_add_clocks_->Update(add_worker));
        --num_waited_add_[add_worker];
      }
    }
  }

private:
  std::unique_ptr<VectorClock> worker_get_clocks_;
  std::unique_ptr<VectorClock> worker_add_clocks_;
  std::vector<int> num_waited_add_;

  MtQueue<MessagePtr> msg_add_cache_;
  MtQueue<MessagePtr> msg_get_cache_;
};

// The BoundedAsyncServer implement logic to support Async SGD training.
// This server assumes:
// - All workers follow blocking Get - Train - Add procedure.
// - A worker can only request a Get or Add at a time.
// - All workers have the same number of iterations.
// The server satisfying the following rules:
// - All workers have a corresponding clock with initial value of 0.
// - When a worker i commits an ADD, it will step its corresponding clock.
//   That is clock[i] += 1.
// - The slowest and fastest workers must be <= MV_CONFIG_basync_clock
//   apart.
// The following condition are promised:
// - A worker Get will see effects of its own Add.
// - A worker Get with clock[i] will see effects of other workers Add 
//   with clock < clock[i] - MV_CONFIG_basync_clock.
// - A worker Get with clock[i] may see effects of other workers Add
//   with clock >= clock[i] - MV_CONFIG_basync_clock.
class BoundedAsyncServer : public Server {
public:
  BoundedAsyncServer() : Server() {
    RegisterHandler(MsgType::Server_Finish_Train, std::bind(
      &BoundedAsyncServer::ProcessFinishTrain, this, std::placeholders::_1));
    num_workers_ = Zoo::Get()->num_workers();
    finish_ = false;
    clocks_.resize(num_workers_, 0);
  }

protected:
  void ProcessAdd(MessagePtr& msg) override {
    CHECK(!finish_);
    PrintClock();
    msg_cache_.Push(msg);
    ProcessQueue();
    PrintClock();
  }

  void ProcessGet(MessagePtr& msg) override {
    CHECK(!finish_);
    PrintClock();
    msg_cache_.Push(msg);
    ProcessQueue();
    PrintClock();
  }

  void ProcessFinishTrain(MessagePtr& msg) {
    ProcessQueue();
    CHECK(msg_cache_.Size() == 0);
  }

private:
  int num_workers_;
  bool finish_;
  std::vector<int> clocks_;
  MtQueue<MessagePtr> msg_cache_;

  void PrintClock() {
    std::string rank = "Rank :" + std::to_string(MV_Rank());
    std::string os = rank + " clock, local: ";
    for (auto i : clocks_) { 
      if (i == std::numeric_limits<int>::max()) os += "-1 ";
      else os += std::to_string(i) + " ";
    }
    Log::Debug("%s\n", os.c_str());
  }

  void ProcessQueue() {
    bool changed = true;
    while (changed && msg_cache_.Size() > 0) {
      changed = false;

      for (int i = 0, size = msg_cache_.Size(); i < size; ++ i) {
        MessagePtr msg;
        msg_cache_.Pop(msg);

        int worker = Zoo::Get()->rank_to_worker_id(msg->src());
        int min_clock = clocks_[worker];
        for (auto val: clocks_) min_clock = std::min(min_clock, val);

        if (msg->type() == Request_Get) {
          if (clocks_[worker] > min_clock + MV_CONFIG_basync_clock) {
            msg_cache_.Push(msg);
          } else {
            Server::ProcessGet(msg);
            changed = true;
          }
        } else if (msg->type() == Request_Add) {
          if (clocks_[worker] >= min_clock + MV_CONFIG_basync_clock) {
            msg_cache_.Push(msg);
          } else {
            Server::ProcessAdd(msg);
            ++ clocks_[worker];
            changed = true;
          }
        } else {
          Log::Fatal("Server received unsupported message type %d\n",
            msg->type());
        }

      } // end for

    } // end while
  }
};

Server* Server::GetServer() {
  if (!MV_CONFIG_sync) {
    Log::Info("Create a async server\n");
    return new Server();
  }
  Log::Info("Create a sync server\n");
  return new SyncServer();
}

}  // namespace multiverso
