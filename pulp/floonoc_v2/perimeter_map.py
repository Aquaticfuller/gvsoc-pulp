#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0, see the LICENSE file for
# details.
# SPDX-License-Identifier: Apache-2.0
#
# Perimeter placement of the L2 channels for the L2 NoC endpoint map.
#
# Port of the RTL's hardware/scripts/gen_perimeter_map.py (TeraNoC repo,
# zexin/teranoc_spatz_mshr branch) into a generator-time helper, so the
# model's L2 NoC endpoint table is DERIVED for any legal power-of-two mesh
# instead of hand-tabled per mesh. Self-contained: no RTL-repo access at
# generation time.
#
# Why placement is not a free choice (docs/scaleup/mesh_plan.md §3.6/§13):
# the L2 interleaver keys the bank select on the L1 word-interleave GROUP
# field, so L2 channel G serves group G (groups share channels above 4x4 in
# consecutive-gid order), and where channel G sits on the perimeter IS group
# G's DMA distance. The assignment is therefore a minimum-total-distance
# assignment, NOT a labeling convention. The committed 4x4 numbering is
# provably optimal (8 hops total, avg 0.50) and is the regression gate here:
# the rule reproduces it exactly (see _SELF_CHECK at import).

# floo_pkg::route_direction_e (tie-break order in the interior pass)
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

# The committed 4x4 placement, decoded from terapool_cluster_floonoc_wrapper.sv.
# Regression gate: assign(4, 4, 16) must reproduce this exactly.
COMMITTED_4X4 = {
    0: ((0, 0), WEST),  1: ((0, 1), WEST),  2: ((0, 2), WEST),  3: ((0, 3), WEST),
    4: ((1, 0), SOUTH), 5: ((0, 0), SOUTH), 6: ((0, 3), NORTH), 7: ((1, 3), NORTH),
    8: ((2, 0), SOUTH), 9: ((3, 0), SOUTH), 10: ((3, 3), NORTH), 11: ((2, 3), NORTH),
    12: ((3, 0), EAST), 13: ((3, 1), EAST), 14: ((3, 2), EAST), 15: ((3, 3), EAST),
}


def group_coord(gid, num_y):
    """gid = x*NumY + y — the encoding the address bit-cast implies."""
    return (gid // num_y, gid % num_y)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def perimeter_points(num_x, num_y):
    """Every off-cluster attachment point. Corners contribute two, on different edges."""
    pts = [((0, y), WEST) for y in range(num_y)]
    pts += [((num_x - 1, y), EAST) for y in range(num_y)]
    pts += [((x, 0), SOUTH) for x in range(num_x)]
    pts += [((x, num_y - 1), NORTH) for x in range(num_x)]
    return pts


def default_channels(num_x, num_y):
    """Largest usable channel count: a power of two, at most the perimeter capacity.

    NumL2Banks must be a power of two because ScrambleBits = clog2(NumL2Banks)
    selects a bit field, so capacity is rounded DOWN.
    """
    cap = 2 * (num_x + num_y)
    n = 1 << (cap.bit_length() - 1)
    return min(n, num_x * num_y)


def channel_of(gid, share):
    """Which channel serves this group: consecutive gids (y-adjacent groups)
    share a channel, matching the coarsened L2 interleave
    (axi_width_interleaved = 16 * share  <=>  bank = group >> log2(share))."""
    return gid // share


def assign(num_x, num_y, num_channels=None):
    """Place each L2 channel at the perimeter point nearest the groups it serves.

    Returns (placement, served): placement[ch] = ((x, y), dir) in ROUTER
    coordinates with an edge direction; served[ch] = [gid, ...].
    """
    num_groups = num_x * num_y
    capacity = 2 * (num_x + num_y)
    num_channels = num_channels or default_channels(num_x, num_y)

    if num_channels > capacity:
        raise ValueError(f"{num_channels} channels exceeds the {capacity} perimeter "
                         f"attachment points of a {num_x}x{num_y} mesh")
    if num_channels & (num_channels - 1):
        raise ValueError(f"NumL2Channels ({num_channels}) must be a power of two — "
                         "ScrambleBits = clog2(NumL2Banks) selects a bit field")
    if num_groups % num_channels:
        raise ValueError(f"{num_groups} groups do not divide evenly into "
                         f"{num_channels} channels")
    share = num_groups // num_channels
    if share & (share - 1):
        raise ValueError(f"sharing factor {share} must be a power of two, so that "
                         "bank = group >> log2(share) is a bit-field select")

    served = {}
    for gid in range(num_groups):
        served.setdefault(channel_of(gid, share), []).append(gid)

    free = set(perimeter_points(num_x, num_y))
    placement, interior = {}, []
    for ch in range(num_channels):
        # The representative is the lowest-numbered group the channel serves.
        # At share == 1 this is the channel's own group, so the edge rule
        # reproduces the committed 4x4 placement exactly.
        x, y = group_coord(served[ch][0], num_y)
        if x == 0:
            pt = ((0, y), WEST)
        elif x == num_x - 1:
            pt = ((num_x - 1, y), EAST)
        elif y == 0:
            pt = ((x, 0), SOUTH)
        elif y == num_y - 1:
            pt = ((x, num_y - 1), NORTH)
        else:
            interior.append(ch)
            continue
        if pt not in free:            # already taken by an earlier channel
            interior.append(ch)
            continue
        placement[ch] = pt
        free.discard(pt)

    # Whatever is left: give each channel the free point minimising the TOTAL
    # distance to every group it serves, not just to its representative.
    for ch in interior:
        pos = [group_coord(g, num_y) for g in served[ch]]
        pt = min(sorted(free),
                 key=lambda q: (sum(manhattan(h, q[0]) for h in pos), q[1], q[0]))
        placement[ch] = pt
        free.discard(pt)

    return improve(placement, served, num_y), served


def improve(placement, served, num_y):
    """Swap two channels' points whenever that lowers total distance (2-opt).

    The edge rule alone is optimal when every group has its own channel, but
    drifts once channels are shared. Deterministic, fixpoint in a few passes.
    Leaves the 4x4 placement untouched (already optimal -> no swap improves it),
    which is what keeps the regression gate meaningful.
    """
    pos = {c: [group_coord(g, num_y) for g in gs] for c, gs in served.items()}

    def cost(c, pt):
        return sum(manhattan(h, pt[0]) for h in pos[c])

    chans = sorted(placement)
    improved = True
    while improved:
        improved = False
        for i in range(len(chans)):
            for j in range(i + 1, len(chans)):
                a, b = chans[i], chans[j]
                pa, pb = placement[a], placement[b]
                if cost(a, pa) + cost(b, pb) > cost(a, pb) + cost(b, pa):
                    placement[a], placement[b] = pb, pa
                    improved = True
    return placement


def periph_channel(placement, periph_at=((0, 0), SOUTH)):
    """The L2 channel sharing the periph router's perimeter point (5 at 4x4,
    13 at 8x8). None when no channel lands there (e.g. 2x2) — the caller then
    keeps the degenerate no-shared-channel demux behaviour."""
    return next((g for g, v in placement.items() if v == periph_at), None)


def to_ni_coord(pt, num_x, num_y):
    """RTL edge point ((x,y), dir) -> L2_noc NI mesh coordinate.

    The model's L2 NoC places NIs one position outside the router grid:
    West(x,y)->(0,y+1), East(nx-1,y)->(nx+1,y+1), South(x,0)->(x+1,0),
    North(x,ny-1)->(x+1,ny+1). Verified against the committed 2x2/4x4 tables.
    """
    (x, y), d = pt
    if d == WEST:
        return (0, y + 1)
    if d == EAST:
        return (num_x + 1, y + 1)
    if d == SOUTH:
        return (x + 1, 0)
    return (x + 1, num_y + 1)  # NORTH


def placement_as_ni(placement, num_x, num_y):
    """{channel: (ni_x, ni_y)} for teranoc_system's o_WIDE_BIND/o_MAP tables."""
    return {ch: to_ni_coord(pt, num_x, num_y) for ch, pt in placement.items()}


def _self_check():
    """Import-time gate: assign(4,4,16) must reproduce the committed 4x4
    placement exactly, or every calibrated 4x4 number silently changes."""
    placement, _served = assign(4, 4, 16)
    bad = [g for g in COMMITTED_4X4 if placement.get(g) != COMMITTED_4X4[g]]
    if bad:
        raise AssertionError(f"perimeter_map: 4x4 placement diverges from the "
                             f"committed RTL mapping at channels {bad}")


_self_check()
