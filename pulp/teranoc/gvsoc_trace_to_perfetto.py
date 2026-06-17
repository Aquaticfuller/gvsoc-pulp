#!/usr/bin/env python3
#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Description: Convert GVSoC TeraNoC traces to Perfetto protobuf traces.
# Author: Yinrong Li (ETH Zurich) (yinrli@student.ethz.ch)
#
"""gvsoc_trace_to_perfetto.py -- GVSoC teranoc --trace -> Perfetto protobuf.

Converts the per-component traces GVSoC emits with ``--trace`` into a Perfetto
*protobuf* trace (``.perfetto-trace``) that loads at https://ui.perfetto.dev,
with a real nested track tree built from TrackDescriptor ``parent_uuid`` -- the
same organization as TeraNoC's ``perf_export.py``.

One converter consumes several ``--trace`` files at once and dispatches each
line on its trace-name leaf in the ``[<path>/<name>]`` token:

  * ``insn``         -> the per-core instruction lane (root ``1 Core``)
  * ``l1_noc_itf``   -> the L1-NoC remote-access lane (root ``2 NoC``)

It is the base we grow the richer metrics on (NoC packets+flows, TCDM banks,
RedMulE FSM/CPI), one trace family at a time.

Requires the ``perfetto`` package (``pip install perfetto``).

Producing the input
-------------------
Run the sim with the traces routed to files (absolute paths -- the run cwd is
the ``--work-dir``; ``insn`` needs ``--trace-level=7``)::

    ./install/bin/gvsoc --target=teranoc --binary <ABS elf> \\
        --work-dir <ABS wk> --config-opt system_config=tensorpool64_noc \\
        --trace-level=7 \\
        "--trace=.*/pe[0-9]*/insn:<ABS>/insn.log" \\
        "--trace=.*l1_noc_itf/trace:<ABS>/noc.log" run

  * insn line:  ``<time_ps>: <cyc>: [<core>/insn] [<func>:<line>] <mode> <pc> <mnem> <ops>``
  * l1_noc_itf: ``... L1_NocItf: core_req port: N addr: 0x.. size: N opcode: N``  (accepted iff
                immediately followed by) ``... core_req translate to noc_req dest_x: X dest_y: Y``

Output
------
    python3 gvsoc_trace_to_perfetto.py insn.log noc.log -o trace.perfetto-trace
    # then open trace.perfetto-trace at ui.perfetto.dev

Track tree (numbered roots so Perfetto's lexicographic root ordering keeps the
lanes in flow order; group keeps its x,y coordinate; children ordered
numerically so e.g. Tile 10 follows Tile 2):

  ``1 Core`` > ``Group X,Y`` > ``Tile N`` > ``Core C``  (insn slices + IPC counter)
  ``2 NoC``  > ``Group X,Y`` > ``Tile N`` > ``Core C``  per-port handshake STATE
                                                         slices (idle/stall/read/write)
                                                         + ``outstanding`` / ``latency``
                                                         counters
  ``3 RedMulE`` > ``Group X,Y`` > ``Tile N``  > ``state`` run-length FSM lane
                                              (PRELOAD/ROUTINE/STORING/FINISHED)
                                              + ``w_out`` / ``compute_pending`` counters
  ``4 TCDM`` > ``Group X,Y`` > ``Tile N`` > ``Bank K``  per-bank access STATE slices
                                              (idle/read/write) + ``accesses`` per-window
                                              utilization counter

With ``--flows``, each ``2 NoC`` port also gets a ``packets`` child lane of instant
req/resp markers connected by click-to-follow flow arrows (one flow per round-trip,
req<->resp paired by IoReq pointer).

``--elf`` adds function names via ``addr2line``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent

# Single producer, absolute timestamps -> one trusted packet sequence.
SEQ = 1
# TrackDescriptor.ChildTracksOrdering.EXPLICIT: order children by sibling_order_rank.
CHILD_ORDER_EXPLICIT = 3
# Path components that form the visible hierarchy (drop teranoc_system/cluster/...).
_HIER_PREFIXES = ("group_", "tile_", "pe")

# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------
# The path token is ANSI-coloured; strip it before parsing.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# "<time_ps>: <cycles>: [<path>] <rest>"  -- same shape gvsoc_analyze_insn parses.
_LINE = re.compile(r"\s*(\d+):\s*(-?\d+):\s*\[([^\]]*)\]\s*(.*)")
# GVSoC prints the PC bare (PRIxFULLREG = "%8.8x"/"%16.16x"), e.g. "80000000" -- no
# "0x" prefix -- and an immediate like 0x5000 keeps its prefix, so accept both forms.
_PC = re.compile(r"^(?:0x)?[0-9a-fA-F]{2,16}$")
# Privilege-mode chars emitted by iss_trace_get_mode(): U/S/H/M (' ' when invalid).
_MODES = frozenset("USHM")

# l1_noc_itf tap (the L1 NoC): a local core (req port 0..3) issues a remote TCDM
# request.  handle_core_req() logs 'core_req port: ... req: <ptr>' on entry, then --
# only on acceptance -- 'core_req translate to noc_req dest_x/dest_y' as the very
# next line (a denied req returns before it).  So a 'port' line immediately
# followed by a 'dest' line is one ACCEPTED request from that core to that group.
# The response returns to the SAME itf as 'noc_resp translate to core_resp ...
# req: <ptr>' carrying the SAME core-IoReq pointer (threaded via FlooNoc REQ_BURST),
# so we pair req<->resp by (tile, ptr) FIFO into a true round-trip [req .. resp].
_NOC_CORE_REQ = re.compile(
    r"L1_NocItf: core_req port: (\d+) addr: (0x[0-9a-fA-F]+) size: (\d+) "
    r"opcode: (\d+) req: (0x[0-9a-fA-F]+)")
_NOC_REQ_DEST = re.compile(
    r"L1_NocItf: core_req translate to noc_req dest_x: (\d+) dest_y: (\d+)")
_NOC_RESP_REQ = re.compile(
    r"L1_NocItf: noc_resp translate to core_resp addr: 0x[0-9a-fA-F]+ "
    r"size: \d+ opcode: \d+ req: (0x[0-9a-fA-F]+)")
# l1_noc_itf per-port STATE markers (the RTL idle/stall/read/write port lane).
# The SOURCE request master noc_req_msts[port] is in exactly one state per cycle:
# read/write = a remote-TCDM transfer issued this cycle; stall = NoC back-pressure
# until the matching unstall (grant).  Idle is the implicit gap.  Non-crossing.
_NOC_STATE = re.compile(r"L1_NocItf: port (\d+) state (read|write|stall|unstall)")

# TCDM L1 bank tap (generic Memory model, instance .../tile_N/l1/tcdm_bankK): the
# model logs one line per serviced access.  A bank is single-port (latency 1) so it
# services one access at a time -> a per-bank read/write run-length lane never crosses.
_BANK = re.compile(
    r"Memory access \(offset: (0x[0-9a-fA-F]+), size: (0x[0-9a-fA-F]+), "
    r"is_write: (\d+)")
_BANK_ID = re.compile(r"tcdm_bank(\d+)")

# LightRedmule tap: the FSM tags each line with its state -- [Preload], [Storing],
# [FINISHED], [ACKNOWLEDGE], or [ROUTINE-ijk: i-j-k] (the GEMM tile being computed).
# [Resp] lines carry the response backpressure counters (outstanding writes +
# whether a compute is waiting on them).  States never overlap -> a run-length lane.
_RM_ROUTINE = re.compile(r"\[LightRedmule\]\[ROUTINE-ijk: (\d+)-(\d+)-(\d+)\]")
_RM_SIMPLE  = re.compile(r"\[LightRedmule\]\[(Preload|Storing|FINISHED|ACKNOWLEDGE)\]")
_RM_RESP    = re.compile(
    r"\[LightRedmule\]\[Resp\] slot=\d+ instr=\d+ "
    r"\(w_out=(\d+), compute_pending=(\d+)\)")


class Insn:
    __slots__ = ("t_ps", "cycle", "core", "mode", "pc", "disasm")

    def __init__(self, t_ps, cycle, core, mode, pc, disasm):
        self.t_ps = t_ps
        self.cycle = cycle
        self.core = core
        self.mode = mode
        self.pc = pc
        self.disasm = disasm


def iter_records(path: str, max_lines: Optional[int] = None):
    """Yield (t_ps, cyc, path_token, rest) for every well-formed trace line
    (ANSI stripped). main() dispatches on the path's trace-name leaf."""
    with open(path, "r", errors="replace") as f:
        for i, raw in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            m = _LINE.match(_ANSI.sub("", raw))
            if not m:
                continue
            t_ps, cyc, brk, rest = m.groups()
            yield int(t_ps), int(cyc), brk.strip(), rest


def trace_leaf(path_token: str) -> str:
    """Last path component, e.g. '.../pe0/insn' -> 'insn'."""
    return path_token.rsplit("/", 1)[-1]


def parse_insn_body(rest: str):
    """From the post-'[path]' text of an ``insn`` line, return (mode, pc_0x,
    disasm) or None.  The "<func>:<line>" debug prefix is only present with
    symbols, so scan for the first privilege-mode char immediately followed by a
    hex PC -- that pair is the real instruction, skipping any debug prefix."""
    toks = rest.split()
    for j in range(len(toks) - 1):
        if toks[j] in _MODES and _PC.match(toks[j + 1]):
            pc = toks[j + 1]
            if not pc.startswith("0x"):         # normalise for addr2line / display
                pc = "0x" + pc
            return toks[j], pc, " ".join(toks[j + 2:])
    return None


def split_hier(path: str) -> List[str]:
    """Ordered hierarchy tokens for a component path, e.g.
    '.../group_3_1/tile_6/pe2' -> ['group_3_1', 'tile_6', 'pe2'] and
    '.../group_3_1/tile_6/l1_noc_itf' -> ['group_3_1', 'tile_6'].
    Falls back to the leaf component when the path has no group/tile/pe levels."""
    parts = [p for p in path.split("/") if p]
    keep = [p for p in parts if p.startswith(_HIER_PREFIXES)]
    return keep or [parts[-1] if parts else path]


def mnemonic(disasm: Optional[str]) -> str:
    return disasm.split(None, 1)[0] if disasm else "?"


class Addr2Line:
    """Batched addr2line -> {pc: 'func'} (best effort; no-op without --elf)."""

    def __init__(self, elf: Optional[str]):
        self.elf = elf
        self.cache: Dict[str, str] = {}

    def resolve(self, pcs: List[str]) -> None:
        if not self.elf:
            return
        todo = sorted({p for p in pcs if p not in self.cache})
        for i in range(0, len(todo), 4000):           # batch like perf_export.py
            chunk = todo[i: i + 4000]
            try:
                out = subprocess.run(
                    ["addr2line", "-f", "-e", self.elf, *chunk],
                    capture_output=True, text=True, check=False,
                ).stdout.splitlines()
            except FileNotFoundError:
                self.elf = None
                return
            # addr2line -f emits two lines per address: function, then file:line
            for j, pc in enumerate(chunk):
                func = out[2 * j].strip() if 2 * j < len(out) else "??"
                self.cache[pc] = func

    def func(self, pc: str) -> str:
        return self.cache.get(pc, "")


# ---------------------------------------------------------------------------
# Perfetto protobuf emission (organization mirrors TeraNoC perf_export.py)
# ---------------------------------------------------------------------------
class Uuids:
    """Deterministic, collision-free uuid allocator keyed by a name tuple."""

    def __init__(self):
        self._n = 0
        self._m: Dict[Tuple, int] = {}
        self.seen: set = set()

    def get(self, key: Tuple) -> int:
        if key not in self._m:
            self._n += 1
            self._m[key] = self._n
        return self._m[key]


def add_track(builder, uuid, name, parent=None, counter=False,
              child_order=None, order_rank=None):
    """Emit one TrackDescriptor (a node of the track tree)."""
    pkt = builder.add_packet()
    td = pkt.track_descriptor
    td.uuid = uuid
    td.name = name
    if parent is not None:
        td.parent_uuid = parent
    if counter:
        td.counter.SetInParent()
    if child_order is not None:
        td.child_ordering = child_order
    if order_rank is not None:
        td.sibling_order_rank = order_rank
    return uuid


def _emit_flow_annos(ev, annos, flow_id, terminating):
    """Shared tail for slice-begin / instant events: attach the flow id (to
    ``flow_ids`` to start/continue a chain, or ``terminating_flow_ids`` ONLY on the
    last hop -- never both, else Perfetto's FlowTracker draws a self-loop) and the
    debug annotations."""
    if flow_id is not None:
        (ev.terminating_flow_ids if terminating else ev.flow_ids).append(flow_id)
    for k, v in (annos or []):
        if v is None or v == "":
            continue
        da = ev.debug_annotations.add()
        da.name = k
        da.string_value = str(v)


def emit_slice_begin(builder, ts_ns, track_uuid, name, annos=None,
                     flow_id=None, terminating=False):
    pkt = builder.add_packet()
    pkt.timestamp = int(ts_ns)
    pkt.trusted_packet_sequence_id = SEQ
    ev = pkt.track_event
    ev.type = TrackEvent.TYPE_SLICE_BEGIN
    ev.track_uuid = track_uuid
    ev.name = name
    _emit_flow_annos(ev, annos, flow_id, terminating)


def emit_instant(builder, ts_ns, track_uuid, name, annos=None,
                 flow_id=None, terminating=False):
    """A zero-duration marker (TYPE_INSTANT) -- used for transaction-flow endpoints.
    Instants cannot overlap/cross (no duration), so many concurrent in-flight remote
    accesses render cleanly and the flow ARROW (not a span) carries the round-trip."""
    pkt = builder.add_packet()
    pkt.timestamp = int(ts_ns)
    pkt.trusted_packet_sequence_id = SEQ
    ev = pkt.track_event
    ev.type = TrackEvent.TYPE_INSTANT
    ev.track_uuid = track_uuid
    ev.name = name
    _emit_flow_annos(ev, annos, flow_id, terminating)


def emit_slice_end(builder, ts_ns, track_uuid):
    pkt = builder.add_packet()
    pkt.timestamp = int(ts_ns)
    pkt.trusted_packet_sequence_id = SEQ
    ev = pkt.track_event
    ev.type = TrackEvent.TYPE_SLICE_END
    ev.track_uuid = track_uuid


def emit_counter(builder, ts_ns, track_uuid, value):
    pkt = builder.add_packet()
    pkt.timestamp = int(ts_ns)
    pkt.trusted_packet_sequence_id = SEQ
    ev = pkt.track_event
    ev.type = TrackEvent.TYPE_COUNTER
    ev.track_uuid = track_uuid
    ev.double_counter_value = float(value)


def level_info(token: str, ny: int = 1) -> Tuple[str, int]:
    """Map a path token to (display name, numeric sibling rank).

    Mirrors the reference perf_export.py "Group/Tile/Core" naming.  The group is
    labelled ``Group <id> (x,y)`` -- both the flat group id and its 2-D mesh
    coordinate -- where the id follows teranoc's convention
    ``group_id = x * nb_y_groups + y`` (teranoc_cluster.py); ``ny`` is
    nb_y_groups (see --ny).  Ranks give a NATURAL numeric ordering so that, with
    EXPLICIT child ordering, tile_10 follows tile_2 instead of preceding it (the
    lexicographic-by-name default mis-sorts two-digit indices)."""
    nums = [int(x) for x in re.findall(r"\d+", token)]
    if token.startswith("group_"):
        if len(nums) >= 2:                       # group_X_Y
            gid = nums[0] * ny + nums[1]
            return f"Group {gid} ({nums[0]},{nums[1]})", gid
        return (f"Group {nums[0]}", nums[0]) if nums else ("Group", 0)
    if token.startswith("tile_"):
        return (f"Tile {nums[0]}", nums[0]) if nums else ("Tile", 0)
    if token.startswith("pe"):
        return (f"Core {nums[0]}", nums[0]) if nums else ("Core", 0)
    return token, (nums[0] if nums else 0)


def ensure_hierarchy(builder, uu: Uuids, tokens: List[str], ny: int,
                     root_name: str = "1 Core", root_key: str = "root") -> int:
    """Declare a numbered root (e.g. '1 Core', '2 NoC') and each group/tile
    container exactly once under it, with EXPLICIT numeric child ordering;
    return the leaf (deepest) track uuid.  ``root_key`` keeps each root's
    group/tile subtree in its own uuid namespace so the lanes never collide."""
    key: Tuple = (root_key,)
    root = uu.get(key)
    if key not in uu.seen:
        add_track(builder, root, root_name, child_order=CHILD_ORDER_EXPLICIT)
        uu.seen.add(key)
    parent = root
    for tok in tokens:
        name, rank = level_info(tok, ny)
        key = key + (tok,)
        u = uu.get(key)
        if key not in uu.seen:
            add_track(builder, u, name, parent=parent, order_rank=rank,
                      child_order=CHILD_ORDER_EXPLICIT)
            uu.seen.add(key)
        parent = u
    return parent


def _ps_to_ns(t_ps: int) -> float:
    # GVSoC trace timestamps are picoseconds; Perfetto wants nanoseconds.
    return t_ps / 1000.0


def emit_core(builder, uu, core, insns, a2l, window_ns, ny):
    """Emit one core's leaf track: instruction slices + an IPC counter child."""
    tokens = split_hier(core)
    leaf = ensure_hierarchy(builder, uu, tokens, ny)
    ipc_uuid = add_track(builder, uu.get(("ipc",) + tuple(tokens)), "IPC",
                         parent=leaf, counter=True, order_rank=0)

    # Instruction slices: each insn spans [t_k, t_{k+1}] (gap to next retirement).
    for k, ins in enumerate(insns):
        emit_slice_begin(builder, _ps_to_ns(ins.t_ps), leaf, mnemonic(ins.disasm),
                         annos=[("pc", ins.pc), ("cycle", ins.cycle),
                                ("disasm", ins.disasm), ("func", a2l.func(ins.pc))])
        nxt = insns[k + 1].t_ps if k + 1 < len(insns) else ins.t_ps
        emit_slice_end(builder, _ps_to_ns(nxt), leaf)

    # IPC counter: one sample per completed window (insns retired / cycles elapsed).
    if insns:
        win_ps = window_ns * 1000.0
        w0 = insns[0].t_ps
        w_end = w0 + win_ps
        n = 0
        cyc0 = insns[0].cycle
        for ins in insns:
            if ins.t_ps >= w_end:
                dcyc = max(ins.cycle - cyc0, 1)
                emit_counter(builder, _ps_to_ns(w0), ipc_uuid, round(n / dcyc, 3))
                while ins.t_ps >= w_end:
                    w0, w_end = w_end, w_end + win_ps
                n, cyc0 = 0, ins.cycle
            n += 1


def emit_noc(builder, uu, tile_path, ev, ny, period_ps, flows=False):
    """One l1_noc_itf instance -> under '2 NoC', per core-port (0..N): a per-port
    handshake-STATE slice lane (the RTL ``idle/stall/read/write`` vocabulary) on the
    ``Core <port>`` track itself, PLUS ``outstanding`` and ``latency`` counter
    children -- mirroring the RTL flow, which keeps state slices and derived counters
    on the same port track.

    State (from the source master noc_req_msts[port]): ``read``/``write`` = one
    1-cycle remote-TCDM transfer issued this cycle; ``stall`` = a span from a NoC
    back-pressure to its grant; idle = the implicit gap.  A source port is in exactly
    ONE state per cycle, so these slices NEVER cross -- unlike the deeply out-of-order
    [req..resp] round-trip spans (tens in flight), which is why the round-trip view is
    rendered as the counters, not as slices.  Req/resp are paired by the core-IoReq
    pointer (threaded via REQ_BURST) per ptr in FIFO order for the counters."""
    tokens = split_hier(tile_path)                  # [group_X_Y, tile_N]
    tile_track = ensure_hierarchy(builder, uu, tokens, ny,
                                  root_name="2 NoC", root_key="noc")
    resps = defaultdict(deque)                       # ptr -> FIFO queue of (t_ps, cyc)
    for ptr, t_ps, cyc in sorted(ev["resp"], key=lambda r: r[1]):
        resps[ptr].append((t_ps, cyc))
    # per core-port: (req_t_ps, req_cyc, resp_t_ps, resp_cyc, paired, addr, dx, dy)
    per_core: Dict[int, list] = defaultdict(list)
    for ptr, rt_ps, rcyc, port, addr, size, op, dx, dy in sorted(
            ev["req"], key=lambda r: r[1]):
        q = resps.get(ptr)
        # Zero-slack guard (RTL SKILL section 4): a response can never precede its own
        # request, so drop any earlier same-pointer resp before pairing.  Such an orphan
        # appears when the capture missed a request (e.g. an fsm-reissued, NoC-stalled
        # one) but kept its response; since the IoReq* is pooled/reused, an unguarded
        # FIFO could otherwise bind THIS request to that neighbour's response.
        if q:
            while q and q[0][0] < rt_ps:
                q.popleft()
        if q:
            st_ps, scyc, paired = (*q.popleft(), True)
        else:
            st_ps, scyc, paired = rt_ps, rcyc, False     # resp past trace window
        per_core[port].append((rt_ps, rcyc, st_ps, scyc, paired, addr, dx, dy))
    ports = sorted(set(per_core) | {p for p, *_ in ev["state"]})
    for port in ports:
        lane = add_track(builder, uu.get(("noc", *tokens, port)),
                         f"Core {port}", parent=tile_track, order_rank=port)
        # Per-port handshake-state slices on the lane track (idle dropped = gap).
        # busy (read/write) = 1-cycle transfers (coalesced when abutting); stall =
        # span to the matching grant.  Merge into ONE start-sorted pass so adjacent
        # busy/stall slices sharing a boundary ns don't mis-nest.
        marks = sorted((cyc, tps, kind)
                       for p, cyc, tps, kind in ev["state"] if p == port)
        spans = []                                   # (start_ps, end_ps, name)
        busy = [(c, t, k) for c, t, k in marks if k in ("read", "write")]
        i = 0
        while i < len(busy):
            c0, t0, nm = busy[i]
            end = c0 + 1
            j = i + 1
            while j < len(busy) and busy[j][2] == nm and busy[j][0] == end:
                end = busy[j][0] + 1
                j += 1
            spans.append((t0, t0 + (end - c0) * period_ps, nm))
            i = j
        unstalls = sorted(t for c, t, k in marks if k == "unstall")
        ui = 0
        for st_ps in sorted(t for c, t, k in marks if k == "stall"):
            while ui < len(unstalls) and unstalls[ui] < st_ps:
                ui += 1
            en_ps = unstalls[ui] if ui < len(unstalls) else st_ps + period_ps
            spans.append((st_ps, en_ps, "stall"))
            ui += 1
        for s_ps, e_ps, nm in sorted(spans):
            emit_slice_begin(builder, _ps_to_ns(s_ps), lane, nm, annos=[("state", nm)])
            emit_slice_end(builder, _ps_to_ns(max(e_ps, s_ps + period_ps)), lane)
        out_uuid = add_track(builder, uu.get(("noc_out", *tokens, port)),
                             "outstanding", parent=lane, counter=True, order_rank=0)
        lat_uuid = add_track(builder, uu.get(("noc_lat", *tokens, port)),
                             "latency", parent=lane, counter=True, order_rank=1)
        # outstanding: +1 at issue, -1 at completion; +1 before -1 at a shared ts.
        deltas = []
        for rt_ps, rcyc, st_ps, scyc, paired, *_ in per_core[port]:
            deltas.append((rt_ps, 0))                    # 0 sorts before 1 -> +1 first
            if paired:
                deltas.append((st_ps, 1))
        deltas.sort()
        cur = 0
        for ts_ps, kind in deltas:
            cur += 1 if kind == 0 else -1
            emit_counter(builder, _ps_to_ns(ts_ps), out_uuid, cur)
        # latency: round-trip cycles, sampled when each response lands.
        for rt_ps, rcyc, st_ps, scyc, paired, *_ in sorted(
                (r for r in per_core[port] if r[4]), key=lambda r: r[2]):
            emit_counter(builder, _ps_to_ns(st_ps), lat_uuid, scyc - rcyc)
        # --flows: a 'packets' child lane of instant req/resp markers, one flow per
        # paired (req,resp).  Instants can't cross (no duration); the round-trip is the
        # flow ARROW.  No buffer/sort needed (RTL gotcha 5): each flow id is unique to a
        # single 2-endpoint flow and the req instant is emitted before its resp instant
        # in one loop iteration, so insertion order == causal order even at a shared ts.
        if flows:
            pkt_lane = add_track(builder, uu.get(("noc_pkt", *tokens, port)),
                                 "packets", parent=lane, order_rank=2)
            for rt_ps, rcyc, st_ps, scyc, paired, addr, dx, dy in per_core[port]:
                gid = dx * ny + dy
                fid = uu.get(("flow", *tokens, port, rcyc)) if paired else None
                emit_instant(builder, _ps_to_ns(rt_ps), pkt_lane, f"req->G{gid}",
                             annos=[("addr", addr), ("dest", f"G{gid} ({dx},{dy})"),
                                    ("req_cyc", rcyc),
                                    ("lat_cyc", scyc - rcyc if paired else None)],
                             flow_id=fid)
                if paired:
                    emit_instant(builder, _ps_to_ns(st_ps), pkt_lane, "resp",
                                 annos=[("addr", addr), ("lat_cyc", scyc - rcyc),
                                        ("resp_cyc", scyc)],
                                 flow_id=fid, terminating=True)


def emit_bank(builder, uu, bank_path, accesses, ny, period_ps, window_ns):
    """One TCDM L1 Memory bank -> under '4 TCDM' > Group > Tile > 'Bank K': a per-bank
    access STATE lane (idle/read/write) + an ``accesses`` per-window utilization counter.

    A bank is single-port (latency 1), so it services one access at a time and the
    state is one-at-a-time -> non-crossing.  We coalesce by CYCLE (not per access): a
    cycle with any access is busy [c, c+1], named ``read``/``write`` (or ``rw`` if a
    conflict put both on one cycle), abutting same-named cycles merged into a run.
    Coalescing by cycle keeps the lane non-crossing even if the arbiter ever lets two
    accesses land on the bank the same cycle (the conflict shows up in the counter)."""
    tokens = split_hier(bank_path)                  # [group_X_Y, tile_N]
    m = _BANK_ID.search(bank_path)
    bank_id = int(m.group(1)) if m else 0
    tile_track = ensure_hierarchy(builder, uu, tokens, ny,
                                  root_name="4 TCDM", root_key="tcdm")
    lane = add_track(builder, uu.get(("tcdm", *tokens, bank_id)),
                     f"Bank {bank_id}", parent=tile_track, order_rank=bank_id)
    # Per cycle: earliest t_ps + the set of access kinds seen that cycle.
    by_cyc: Dict[int, list] = {}
    for cyc, t_ps, addr, size, isw in accesses:
        nm = "write" if isw else "read"
        if cyc not in by_cyc:
            by_cyc[cyc] = [t_ps, set()]
        by_cyc[cyc][0] = min(by_cyc[cyc][0], t_ps)
        by_cyc[cyc][1].add(nm)
    cyc_state = []                                   # (cyc, t_ps, name)
    for cyc in sorted(by_cyc):
        t_ps, kinds = by_cyc[cyc]
        cyc_state.append((cyc, t_ps, "rw" if len(kinds) > 1 else next(iter(kinds))))
    # Coalesce abutting same-named cycles into one busy slice (idle = gap).
    i = 0
    while i < len(cyc_state):
        c0, t0, nm = cyc_state[i]
        end = c0 + 1
        j = i + 1
        while j < len(cyc_state) and cyc_state[j][2] == nm and cyc_state[j][0] == end:
            end = cyc_state[j][0] + 1
            j += 1
        emit_slice_begin(builder, _ps_to_ns(t0), lane, nm, annos=[("state", nm)])
        emit_slice_end(builder, _ps_to_ns(t0 + (end - c0) * period_ps), lane)
        i = j
    # accesses-per-window utilization counter (bank hotspot signal).
    rate_uuid = add_track(builder, uu.get(("tcdm_rate", *tokens, bank_id)),
                          "accesses", parent=lane, counter=True, order_rank=0)
    if accesses:
        win_ps = window_ns * 1000.0
        ts = sorted(t_ps for _, t_ps, *_ in accesses)
        w0 = ts[0]
        w_end = w0 + win_ps
        n = 0
        for t_ps in ts:
            if t_ps >= w_end:
                emit_counter(builder, _ps_to_ns(w0), rate_uuid, n)
                while t_ps >= w_end:
                    w0, w_end = w_end, w_end + win_ps
                n = 0
            n += 1
        emit_counter(builder, _ps_to_ns(w0), rate_uuid, n)


def emit_redmule(builder, uu, rm_path, ev, ny):
    """One LightRedmule instance -> under '3 RedMulE': a run-length FSM ``state``
    lane (PRELOAD/ROUTINE/STORING/FINISHED, ROUTINE annotated with the i-j-k GEMM
    tile) plus ``w_out`` (outstanding writes) and ``compute_pending`` counters from
    the [Resp] line.  Only one FSM state is live at a time, so the lane never
    crosses; the counters expose when compute stalls waiting on stores to drain."""
    tokens = split_hier(rm_path)                    # [group_X_Y, tile_N]
    tile_track = ensure_hierarchy(builder, uu, tokens, ny,
                                  root_name="3 RedMulE", root_key="redmule")
    lane = add_track(builder, uu.get(("rm", *tokens)), "state",
                     parent=tile_track, order_rank=0)
    wout_uuid = add_track(builder, uu.get(("rm_wout", *tokens)), "w_out",
                          parent=tile_track, counter=True, order_rank=1)
    cp_uuid = add_track(builder, uu.get(("rm_cp", *tokens)), "compute_pending",
                        parent=tile_track, counter=True, order_rank=2)
    # Coalesce consecutive same-state lines into runs; each run's end is the next
    # run's start (the last run ends at the final state line).
    states = sorted(ev["state"])                     # (t_ps, label, ijk)
    runs = []                                        # [label, ijk, start_ps, end_ps]
    for t_ps, label, ijk in states:
        if runs and runs[-1][0] == label and runs[-1][1] == ijk:
            continue
        runs.append([label, ijk, t_ps, t_ps])
    for i, run in enumerate(runs):
        run[3] = runs[i + 1][2] if i + 1 < len(runs) else states[-1][0]
    for label, ijk, s_ps, e_ps in runs:
        emit_slice_begin(builder, _ps_to_ns(s_ps), lane, label,
                         annos=[("ijk", ijk)] if ijk else None)
        emit_slice_end(builder, _ps_to_ns(max(e_ps, s_ps + 1)), lane)
    for t_ps, wout, cp in sorted(ev["resp"]):
        emit_counter(builder, _ps_to_ns(t_ps), wout_uuid, wout)
        emit_counter(builder, _ps_to_ns(t_ps), cp_uuid, cp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="+",
                    help="one or more GVSoC --trace files (insn and/or l1_noc_itf)")
    ap.add_argument("-o", "--output", default=None,
                    help="output protobuf (default: <first-trace>.perfetto-trace)")
    ap.add_argument("--elf", default=None, help="ELF for addr2line function names")
    ap.add_argument("--window-ns", type=float, default=1000.0,
                    help="IPC counter window (ns)")
    ap.add_argument("--ny", type=int, default=None,
                    help="nb_y_groups for the flat group id (group_id = x*ny+y); "
                         "default: inferred from the trace (max group-y + 1) -- "
                         "pass it explicitly for a partial trace")
    ap.add_argument("--max-lines", type=int, default=None,
                    help="cap input lines per file (debug)")
    ap.add_argument("--flows", action="store_true",
                    help="emit NoC transaction flows: per-port 'packets' lane of "
                         "instant req/resp markers connected by click-to-follow arrows "
                         "(req<->resp paired by IoReq pointer)")
    args = ap.parse_args()

    for p in args.trace:
        if not Path(p).is_file():
            print(f"ERROR: no such trace file: {p}", file=sys.stderr)
            return 2

    by_core: Dict[str, List[Insn]] = defaultdict(list)
    # noc[tile_path] = {"req":   [(ptr, t_ps, cyc, core_port, addr, size, op, dx, dy)],
    #                   "resp":  [(ptr, t_ps, cyc)],
    #                   "state": [(port, cyc, t_ps, kind)]  kind=read|write|stall|unstall}
    noc: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {"req": [], "resp": [], "state": []})
    cyc_ps_lo = cyc_ps_hi = None    # (cyc, t_ps) extrema -> ps-per-cycle period
    # rm[redmule_path] = {"state": [(t_ps, label, ijk_or_None)],
    #                     "resp":  [(t_ps, w_out, compute_pending)]}
    rm: Dict[str, Dict[str, list]] = defaultdict(lambda: {"state": [], "resp": []})
    # banks[bank_path] = [(cyc, t_ps, addr, size, is_write)]  one serviced access each
    banks: Dict[str, list] = defaultdict(list)

    for p in args.trace:
        pending = None          # last 'core_req port' line, awaiting its 'dest' line
        for t_ps, cyc, brk, rest in iter_records(p, args.max_lines):
            if cyc_ps_lo is None or cyc < cyc_ps_lo[0]:
                cyc_ps_lo = (cyc, t_ps)
            if cyc_ps_hi is None or cyc > cyc_ps_hi[0]:
                cyc_ps_hi = (cyc, t_ps)
            leaf = trace_leaf(brk)
            if leaf == "insn":
                body = parse_insn_body(rest)
                if body:
                    mode, pc, disasm = body
                    core = brk[:-len("/insn")]
                    by_core[core].append(Insn(t_ps, cyc, core, mode, pc, disasm))
                continue
            if "redmule" in brk:
                m = _RM_ROUTINE.search(rest)
                if m:
                    rm[brk]["state"].append(
                        (t_ps, "ROUTINE", f"{m.group(1)}-{m.group(2)}-{m.group(3)}"))
                    continue
                m = _RM_SIMPLE.search(rest)
                if m:
                    rm[brk]["state"].append((t_ps, m.group(1).upper(), None))
                    continue
                m = _RM_RESP.search(rest)
                if m:
                    rm[brk]["resp"].append((t_ps, int(m.group(1)), int(m.group(2))))
                continue
            if "tcdm_bank" in brk:
                m = _BANK.search(rest)
                if m:
                    banks[brk].append((cyc, t_ps, m.group(1),
                                       int(m.group(2), 16), int(m.group(3))))
                continue
            if "l1_noc_itf" not in brk:
                continue
            m = _NOC_CORE_REQ.search(rest)
            if m:
                # a new 'port' line drops any unconsumed pending (it was denied)
                pending = (brk, int(m.group(1)), m.group(2), int(m.group(3)),
                           int(m.group(4)), m.group(5), t_ps, cyc)
                continue
            m = _NOC_REQ_DEST.search(rest)
            if m and pending:
                b, port, addr, size, op, ptr, pt_ps, pcyc = pending
                noc[b]["req"].append((ptr, pt_ps, pcyc, port, addr, size, op,
                                      int(m.group(1)), int(m.group(2))))
                pending = None
                continue
            m = _NOC_RESP_REQ.search(rest)
            if m:
                noc[brk]["resp"].append((m.group(1), t_ps, cyc))
                continue
            m = _NOC_STATE.search(rest)
            if m:
                noc[brk]["state"].append((int(m.group(1)), cyc, t_ps, m.group(2)))

    if not by_core and not noc and not rm and not banks:
        print("ERROR: no insn, l1_noc_itf, redmule or tcdm_bank lines parsed -- check "
              "the --trace regex (need .*/insn, .*l1_noc_itf/trace, .*redmule/trace "
              "and/or .*tcdm_bank.*, with --trace-level=7)", file=sys.stderr)
        return 1

    a2l = Addr2Line(args.elf)
    a2l.resolve([i.pc for ins in by_core.values() for i in ins])

    # nb_y_groups for the flat group id; infer across all parsed paths when absent.
    ny = args.ny
    if ny is None:
        ys = [int(re.findall(r"\d+", t)[1])
              for path in list(by_core) + list(noc) + list(rm) + list(banks)
              for t in split_hier(path)
              if t.startswith("group_") and len(re.findall(r"\d+", t)) >= 2]
        ny = max(ys) + 1 if ys else 1

    # ps-per-cycle, for the 1-cycle width of the NoC busy-state slices.
    period_ps = 1000.0
    if cyc_ps_lo and cyc_ps_hi and cyc_ps_hi[0] != cyc_ps_lo[0]:
        period_ps = (cyc_ps_hi[1] - cyc_ps_lo[1]) / (cyc_ps_hi[0] - cyc_ps_lo[0])

    builder = TraceProtoBuilder()
    uu = Uuids()
    for core in sorted(by_core):
        emit_core(builder, uu, core, by_core[core], a2l, args.window_ns, ny)
    for tile_path in sorted(noc):
        emit_noc(builder, uu, tile_path, noc[tile_path], ny, period_ps, args.flows)
    for rm_path in sorted(rm):
        emit_redmule(builder, uu, rm_path, rm[rm_path], ny)
    for bank_path in sorted(banks):
        emit_bank(builder, uu, bank_path, banks[bank_path], ny, period_ps,
                  args.window_ns)

    out = args.output or (args.trace[0] + ".perfetto-trace")
    with open(out, "wb") as f:
        f.write(builder.serialize())

    n_insn = sum(len(v) for v in by_core.values())
    n_req = sum(len(ev["req"]) for ev in noc.values())
    n_resp = sum(len(ev["resp"]) for ev in noc.values())
    n_noc_state = sum(len(ev["state"]) for ev in noc.values())
    n_rm_state = sum(len(ev["state"]) for ev in rm.values())
    n_bank_acc = sum(len(v) for v in banks.values())
    print(f"cores: {len(by_core)}  instructions: {n_insn}  "
          f"noc-itf: {len(noc)}  noc-reqs: {n_req}  noc-resps: {n_resp}  "
          f"noc-state: {n_noc_state}  period_ps: {period_ps:.0f}  "
          f"banks: {len(banks)}  bank-accesses: {n_bank_acc}  "
          f"flows: {'on' if args.flows else 'off'}  "
          f"redmule: {len(rm)}  rm-state-lines: {n_rm_state}  "
          f"tracks: {uu._n}  ny={ny}{'' if args.ny is not None else ' (inferred)'}")
    print(f"wrote {out}  (open at https://ui.perfetto.dev)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
