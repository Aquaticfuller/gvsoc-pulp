"""Exercise the production MMIO frontend against a delayed splitter sink.

Run with python3 test_dma_frontend.py. The small VP shim lets the real
frontend run without an ELF; it does not implement DMA register behavior.
"""
from pathlib import Path
import os
import subprocess
import tempfile

header = Path(__file__).resolve().parents[2] / 'pulp/teranoc_spatz/dma/teranoc_dma_frontend.hpp'
vp = r'''
#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>
namespace vp {
inline int64_t cycle = 0;
enum { DEBUG };
struct Trace {
  enum { LEVEL_TRACE, LEVEL_INFO };
  template<class... T> void msg(int, const char *, T...) {}
  template<class... T> void fatal(const char *, T...) { std::abort(); }
};
struct Traces { void new_trace(const char *, Trace *, int) {} };
struct Clock { int64_t get_cycles() { return cycle; } };
struct Component;
struct Block {
  Clock clock; Traces traces;
  Block(Component *, std::string) {}
  virtual ~Block() = default;
  virtual void reset(bool) {}
};
struct Component {
  template<class T> void new_slave_port(const char *, T *, Block *) {}
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
io = r'''
#pragma once
#include <vp/vp.hpp>
namespace vp {
enum IoReqStatus { IO_REQ_DONE, IO_REQ_DENIED };
enum { IO_RESP_OK, IO_RESP_INVALID };
struct IoReq {
  uint64_t addr; uint32_t data; bool write; int status = -1;
  int get_size() { return 4; }
  void *get_data() { return &data; }
  uint64_t get_addr() { return addr; }
  bool get_is_write() { return write; }
  void set_resp_status(int s) { status = s; }
};
struct IoSlave {
  IoReqStatus (*handler)(Block *, IoReq *); int retries = 0;
  IoSlave(IoReqStatus (*h)(Block *, IoReq *)): handler(h) {}
  void retry() { retries++; }
};
}
'''
idma = r'''
#pragma once
#include <cstdint>
struct IdmaTransfer {
  uint64_t src, dst, size, src_stride, dst_stride;
  int reps, config, ack_size, nb_bursts; bool bursts_sent;
  IdmaTransfer *parent;
};
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
#include <unordered_map>
#include <vp/itf/io_v2.hpp>
#include <ips/pulp/idma_v2/idma.hpp>
#define private public
#include "teranoc_dma_frontend.hpp"
#undef private
struct Sink : IdmaTransferConsumer {
  bool ready = true; std::deque<IdmaTransfer *> received;
  bool can_accept_transfer() override { return ready; }
  void enqueue_transfer(IdmaTransfer *t) override { received.push_back(t); }
};
int main() {
  vp::Component owner; Sink sink; TeranocDmaFrontend fe(&owner, &sink);
  auto access = [&](uint64_t addr, bool write = false, uint32_t value = 0) {
    vp::IoReq req{addr, value, write};
    auto result = fe.input_itf.handler(&fe, &req);
    assert(result == vp::IO_REQ_DONE && req.status == vp::IO_RESP_OK);
    return req.data;
  };
  auto tick = [&] {
    vp::cycle++;
    fe.issue_event.run(); fe.completion_event.run();
  };
  auto complete = [&] {
    assert(!sink.received.empty());
    auto *t = sink.received.front(); sink.received.pop_front();
    fe.ack_transfer(t);
  };
  auto done = [&] { return access(MMIO_DONE); };
  fe.reset(true); fe.reset(false);
  assert(access(MMIO_STATUS) == 1);
  for (int i = 0; i < 16; i++) { assert(done() == 0); tick(); }
  access(MMIO_SRC, true, 0x80000000);
  access(MMIO_DST, true, 0x10000000);
  access(MMIO_SIZE, true, 64);
  assert(access(MMIO_NEXT_ID) == 0 && done() == 0);
  assert(access(MMIO_STATUS) == 0 && sink.received.empty());
  tick(); assert(sink.received.size() == 1 && done() == 0);
  assert(sink.received.front()->size == 64);
  complete();
  if (teranoc_dma_clocked()) assert(done() == 0);
  tick();
  for (int i = 0; i < 4; i++) assert(done() == 1); // reads are not clear-on-read
  assert(access(MMIO_STATUS) == 1);
  // Launch clears a previous completion. Only the last retirement sets done.
  access(MMIO_NEXT_ID); assert(done() == 0); tick();
  access(MMIO_NEXT_ID); tick(); assert(sink.received.size() == 2);
  complete(); tick(); assert(done() == 0);
  complete(); tick(); assert(done() == 1);
  // A zero-length descriptor also clears done and waits for retirement.
  access(MMIO_SIZE, true, 0);
  access(MMIO_NEXT_ID); assert(done() == 0); tick();
  assert(sink.received.front()->size == 0);
  complete(); tick(); assert(done() == 1);
  // A denied launch must not modify registers or counters before retry.
  sink.ready = false;
  access(MMIO_NEXT_ID); access(MMIO_NEXT_ID);
  vp::IoReq held{MMIO_NEXT_ID, 99, false};
  auto issued = fe.issued;
  assert(fe.input_itf.handler(&fe, &held) == vp::IO_REQ_DENIED);
  assert(fe.issued == issued && done() == 0);
  sink.ready = true; fe.update(); tick();
  assert(fe.input_itf.retries == 1);
  assert(fe.input_itf.handler(&fe, &held) == vp::IO_REQ_DONE && held.data == 0);
  tick(); tick(); assert(sink.received.size() == 3);
  while (!sink.received.empty()) { complete(); tick(); }
  assert(done() == 1);
  // Reset must clear the sticky flag even after successful work.
  fe.reset(true); assert(done() == 0 && access(MMIO_STATUS) == 1);
  access(MMIO_SIZE, true, 64); access(MMIO_NEXT_ID); fe.reset(true);
  tick(); assert(sink.received.empty() && done() == 0);
}
'''
with tempfile.TemporaryDirectory(prefix='dma-frontend-test-') as tmp:
    path = Path(tmp)
    (path / 'vp/itf').mkdir(parents=True)
    (path / 'ips/pulp/idma_v2').mkdir(parents=True)
    (path / 'vp/vp.hpp').write_text(vp)
    (path / 'vp/itf/io_v2.hpp').write_text(io)
    (path / 'ips/pulp/idma_v2/idma.hpp').write_text(idma)
    (path / 'test.cpp').write_text(test)
    subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(path), '-I' + str(header.parent), str(path / 'test.cpp'),
                    '-o', str(path / 'test')], check=True)
    for clocked in ('0', '1'):
        env = dict(os.environ, TERANOC_DMA_CLOCKED=clocked)
        env.pop('TERANOC_DMA_STATS_PATH', None)
        subprocess.run([str(path / 'test')], check=True, env=env)
        print(f'PASS DMA frontend (clocked={clocked}): reset, sticky done, launch, '
              'completion, queued work, zero length, denied launch, reset after work')
