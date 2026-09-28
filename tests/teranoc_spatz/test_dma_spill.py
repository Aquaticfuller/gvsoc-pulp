"""Exercise the production DMA spill against a clock and backpressured sink."""
from pathlib import Path
import subprocess
import tempfile

header = Path(__file__).resolve().parents[2] / 'pulp/teranoc_spatz/dma/idma_transfer_spill.hpp'
vp = r'''
#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace vp {
inline int64_t cycle = 0;
struct Clock { int64_t get_cycles() { return cycle; } };
struct Component {};
struct Block {
  Clock clock;
  Block(Component *, std::string) {}
  virtual ~Block() = default;
  virtual void reset(bool) {}
};
struct ClockEvent {
  Block *block; void (*callback)(Block *, ClockEvent *); int64_t due = -1;
  ClockEvent(Block *b, void (*cb)(Block *, ClockEvent *)): block(b), callback(cb) {}
  bool is_enqueued() { return due >= 0; }
  void enqueue(int delay = 1) { due = cycle + delay; }
  void cancel() { due = -1; }
  void run() { if (due == cycle) { due = -1; callback(block, this); } }
};
}
'''
idma = r'''
#pragma once
struct IdmaTransfer {};
struct IdmaTransferConsumer {
  virtual bool can_accept_transfer() = 0;
  virtual void enqueue_transfer(IdmaTransfer *) = 0;
};
struct IdmaTransferProducer {
  virtual void update() = 0;
  virtual void ack_transfer(IdmaTransfer *) = 0;
};
'''
test = r'''
#include <cassert>
#include <deque>
#include <cstdlib>
#include <ips/pulp/idma_v2/idma.hpp>
#include <vp/vp.hpp>
#define private public
#include "idma_transfer_spill.hpp"
#undef private
struct Producer : IdmaTransferProducer {
  int updates = 0; std::vector<IdmaTransfer *> done;
  void update() override { updates++; }
  void ack_transfer(IdmaTransfer *t) override { done.push_back(t); }
};
struct Consumer : IdmaTransferConsumer {
  bool ready = false; std::vector<IdmaTransfer *> received;
  bool can_accept_transfer() override { return ready; }
  void enqueue_transfer(IdmaTransfer *t) override { received.push_back(t); }
};
int main() {
  vp::Component owner; Producer producer; Consumer consumer;
  IDmaTransferSpill spill(&owner, "spill", &producer, &consumer);
  IdmaTransfer a,b,c;
  spill.enqueue_transfer(&a); spill.enqueue_transfer(&b);
  assert(!spill.can_accept_transfer() && consumer.received.empty());
  vp::cycle=1; spill.event.run(); // denied downstream: retain both descriptors
  assert(consumer.received.empty() && !spill.event.is_enqueued());
  consumer.ready=true; spill.update();
  vp::cycle=2; spill.event.run();
  assert(consumer.received.size()==1 && consumer.received[0]==&a);
  assert(!spill.can_accept_transfer()); // full-at-start credit is not reused
  vp::cycle=3; spill.event.run();
  assert(consumer.received.size()==2 && consumer.received[1]==&b);
  assert(spill.can_accept_transfer() && producer.updates>0);
  spill.ack_transfer(&a); spill.ack_transfer(&b);
  assert(producer.done.empty());
  vp::cycle=4; spill.event.run(); assert(producer.done.size()==1);
  vp::cycle=5; spill.event.run(); assert(producer.done.size()==2);
  spill.enqueue_transfer(&c); spill.reset(true);
  vp::cycle=6; spill.event.run(); assert(consumer.received.size()==2);
  // Two actual RTL cuts cannot collapse into same-cycle forwarding.
  IDmaTransferSpill second(&owner, "second", &producer, &consumer);
  IDmaTransferSpill first(&owner, "first", &producer, &second);
  first.enqueue_transfer(&c);
  vp::cycle=7; first.event.run(); second.event.run();
  assert(consumer.received.size()==2);
  vp::cycle=8; first.event.run(); second.event.run();
  assert(consumer.received.size()==3 && consumer.received.back()==&c);
}
'''
with tempfile.TemporaryDirectory(prefix='dma-spill-test-') as tmp:
    path = Path(tmp)
    (path / 'vp').mkdir()
    (path / 'ips/pulp/idma_v2').mkdir(parents=True)
    (path / 'vp/vp.hpp').write_text(vp)
    (path / 'ips/pulp/idma_v2/idma.hpp').write_text(idma)
    (path / 'test.cpp').write_text(test)
    subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(path), '-I' + str(header.parent), str(path / 'test.cpp'),
                    '-o', str(path / 'test')], check=True)
    subprocess.run([str(path / 'test')], check=True)
print('PASS DMA spill: clocked issue, backpressure, full credit, completion, reset, two cuts')
