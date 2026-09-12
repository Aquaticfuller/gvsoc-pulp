#pragma once

#include <vector>
#include <stdexcept>

namespace teranoc {
// mempool_group_mshr.sv: gen_arb_port/gen_arb_combine. Each port searches
// slots >= alloc_rr_q before wrapping; the base's low bit selects the
// preferred port when both have a candidate for the same bank. Ordering
// requests this way lets the independent per-bank allocation/merge gates
// pick those winners, without letting callback arrival order choose them.
inline std::vector<int> mshr_request_order(int lanes, int ports, int base)
{
    if (lanes <= 0 || ports != 2 || lanes % ports || base < 0 || base >= lanes)
        throw std::invalid_argument("MSHR arbitration requires two active request ports per tile");
    std::vector<int> order;
    order.reserve(lanes);
    for (int rank = 0; rank < ports; ++rank)
    {
        int port = (base + rank) % ports;
        for (int wrap = 0; wrap < 2; ++wrap)
            for (int lane = port; lane < lanes; lane += ports)
                if ((lane >= base) == (wrap == 0)) order.push_back(lane);
    }
    return order;
}
}
