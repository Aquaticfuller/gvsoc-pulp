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
  ``2 NoC``  > ``Group X,Y`` > ``Tile N`` > ``Core C``  (remote-request slices, named by
                                                         destination group)

``--elf`` adds function names via ``addr2line``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
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
# request.  handle_core_req() logs 'core_req port: ...' on entry, then -- only on
# acceptance -- 'core_req translate to noc_req dest_x/dest_y' as the very next
# line (a denied req returns before it).  So a 'port' line immediately followed by
# a 'dest' line is one ACCEPTED remote request from that core to that group.
# (Pairing the response back for true round-trip latency needs a correlation id
# threaded into the trace -- noc_resp carries neither the core nor an id; P2.)
_NOC_CORE_REQ = re.compile(
    r"L1_NocItf: core_req port: (\d+) addr: (0x[0-9a-fA-F]+) size: (\d+) opcode: (\d+)")
_NOC_REQ_DEST = re.compile(
    r"L1_NocItf: core_req translate to noc_req dest_x: (\d+) dest_y: (\d+)")


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


def emit_slice_begin(builder, ts_ns, track_uuid, name, annos=None):
    pkt = builder.add_packet()
    pkt.timestamp = int(ts_ns)
    pkt.trusted_packet_sequence_id = SEQ
    ev = pkt.track_event
    ev.type = TrackEvent.TYPE_SLICE_BEGIN
    ev.track_uuid = track_uuid
    ev.name = name
    for k, v in (annos or []):
        if v is None or v == "":
            continue
        da = ev.debug_annotations.add()
        da.name = k
        da.string_value = str(v)


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


def emit_noc(builder, uu, tile_path, by_port, ny):
    """One l1_noc_itf instance -> under '2 NoC', a per-core lane of accepted
    remote-TCDM requests, each named by its destination group ``->G<gid>`` (so
    requests to the same group share a colour).  Each request spans the gap to
    that core's next request (its remote-access cadence); the Snitch LSU blocks
    on the access, so per-core requests are sequential and the slices never
    overlap.  True round-trip latency (req..resp) arrives in P2 with a
    correlation id in the trace."""
    tokens = split_hier(tile_path)                  # [group_X_Y, tile_N]
    tile_track = ensure_hierarchy(builder, uu, tokens, ny,
                                  root_name="2 NoC", root_key="noc")
    for port in sorted(by_port):
        ct = add_track(builder, uu.get(("noc", *tokens, port)),
                       f"Core {port}", parent=tile_track, order_rank=port)
        reqs = sorted(by_port[port])                 # (t_ps, cyc, addr, size, op, dx, dy)
        for k, (t_ps, cyc, addr, size, op, dx, dy) in enumerate(reqs):
            gid = dx * ny + dy
            emit_slice_begin(builder, _ps_to_ns(t_ps), ct, f"→G{gid}",
                             annos=[("addr", addr), ("size", size), ("opcode", op),
                                    ("dest", f"G{gid} ({dx},{dy})"), ("cycle", cyc)])
            nxt = reqs[k + 1][0] if k + 1 < len(reqs) else t_ps
            emit_slice_end(builder, _ps_to_ns(nxt), ct)


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
    args = ap.parse_args()

    for p in args.trace:
        if not Path(p).is_file():
            print(f"ERROR: no such trace file: {p}", file=sys.stderr)
            return 2

    by_core: Dict[str, List[Insn]] = defaultdict(list)
    # noc[tile_path][core_port] = [(t_ps, cyc, addr, size, opcode, dest_x, dest_y)]
    noc: Dict[str, Dict[int, list]] = defaultdict(lambda: defaultdict(list))

    for p in args.trace:
        pending = None          # last 'core_req port' line, awaiting its 'dest' line
        for t_ps, cyc, brk, rest in iter_records(p, args.max_lines):
            leaf = trace_leaf(brk)
            if leaf == "insn":
                body = parse_insn_body(rest)
                if body:
                    mode, pc, disasm = body
                    core = brk[:-len("/insn")]
                    by_core[core].append(Insn(t_ps, cyc, core, mode, pc, disasm))
                continue
            if "l1_noc_itf" not in brk:
                continue
            m = _NOC_CORE_REQ.search(rest)
            if m:
                # a new 'port' line drops any unconsumed pending (it was denied)
                pending = (brk, int(m.group(1)), m.group(2), int(m.group(3)),
                           int(m.group(4)), t_ps, cyc)
                continue
            m = _NOC_REQ_DEST.search(rest)
            if m and pending:
                b, port, addr, size, op, pt_ps, pcyc = pending
                noc[b][port].append((pt_ps, pcyc, addr, size, op,
                                     int(m.group(1)), int(m.group(2))))
                pending = None

    if not by_core and not noc:
        print("ERROR: no insn or l1_noc_itf lines parsed -- check the --trace regex "
              "(need .*/insn and/or .*l1_noc_itf/trace, with --trace-level=7)",
              file=sys.stderr)
        return 1

    a2l = Addr2Line(args.elf)
    a2l.resolve([i.pc for ins in by_core.values() for i in ins])

    # nb_y_groups for the flat group id; infer across all parsed paths when absent.
    ny = args.ny
    if ny is None:
        ys = [int(re.findall(r"\d+", t)[1])
              for path in list(by_core) + list(noc) for t in split_hier(path)
              if t.startswith("group_") and len(re.findall(r"\d+", t)) >= 2]
        ny = max(ys) + 1 if ys else 1

    builder = TraceProtoBuilder()
    uu = Uuids()
    for core in sorted(by_core):
        emit_core(builder, uu, core, by_core[core], a2l, args.window_ns, ny)
    for tile_path in sorted(noc):
        emit_noc(builder, uu, tile_path, noc[tile_path], ny)

    out = args.output or (args.trace[0] + ".perfetto-trace")
    with open(out, "wb") as f:
        f.write(builder.serialize())

    n_insn = sum(len(v) for v in by_core.values())
    n_noc = sum(len(rl) for bp in noc.values() for rl in bp.values())
    print(f"cores: {len(by_core)}  instructions: {n_insn}  "
          f"noc-itf: {len(noc)}  noc-reqs: {n_noc}  tracks: {uu._n}  "
          f"ny={ny}{'' if args.ny is not None else ' (inferred)'}")
    print(f"wrote {out}  (open at https://ui.perfetto.dev)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
