#include "../../pulp/teranoc_spatz/mshr_request_order.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

// Independent RTL-style oracle: build per-port candidate masks, isolate the
// lowest set bit above the thermometer base (or wrap), then apply port yield.
static int rtl_winner(uint32_t candidates, int base)
{
    int winners[2] = {-1, -1};
    for (int p = 0; p < 2; ++p)
    {
        uint32_t mask = candidates & (p ? 0xaaaaaaaaU : 0x55555555U);
        uint32_t after = mask & (0xffffffffU << base);
        uint32_t chosen = after ? after : mask;
        if (chosen) winners[p] = __builtin_ctz(chosen);
    }
    int preferred = base & 1;
    return winners[preferred] >= 0 ? winners[preferred] : winners[preferred ^ 1];
}

int main()
{
    std::mt19937 random(0x20260912);
    int service[32] = {};
    for (int base = 0; base < 32; ++base)
    {
        auto order = teranoc::mshr_request_order(32, 2, base);
        auto sorted = order;
        std::sort(sorted.begin(), sorted.end());
        for (int lane = 0; lane < 32; ++lane) assert(sorted[lane] == lane);
        ++service[order.front()];
        for (int trial = 0; trial < 10000; ++trial)
        {
            uint32_t candidates = random();
            int winner = -1;
            for (int lane : order)
                if (candidates & (1U << lane)) { winner = lane; break; }
            assert(winner == rtl_winner(candidates, base));
        }
        // A base odd/even port preference can outrank the numerically nearest
        // slot of the other port. Plain cyclic lane traversal is not equivalent.
        assert(order.front() == base);
    }
    for (int count : service) assert(count == 1);
    assert(teranoc::mshr_request_order(32, 2, 2)[1] == 4);
    std::cout << "PASS 320000 candidate sets against RTL port masks/yield; every continuously requesting lane served per rotation\n";
}
