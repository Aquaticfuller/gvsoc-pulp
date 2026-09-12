// Check the model against the RTL selector-register rotation and port bundles.
#include "../../pulp/teranoc_spatz/l1_interconnect/remapper_mapping.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int main()
{
    // Regression witness: lane 8 is slot 1 in interleaved group 0.
    assert(teranoc::remap_port(8, 32, 4, true, 0) == 16);
    assert(teranoc::remap_port(8, 32, 4, true, 0, true) == 2);
    for (int ports : {8, 16, 32, 64})
    for (int batch : {1, 2, 4, 8})
    for (bool interleaved : {false, true})
    {
        std::vector<int> sel(batch);
        for (int i = 0; i < batch; ++i)
            sel[i] = batch == 1 ? 0 : (i < batch/2 ? 2*i : 2*i-batch+1);
        int groups = ports / batch;
        for (int cycle = 0; cycle < 2*batch; ++cycle)
        {
            std::vector<bool> seen(ports, false);
            for (int g = 0; g < groups; ++g)
            for (int i = 0; i < batch; ++i)
            {
                int input = interleaved ? g+i*groups : g*batch+i;
                int expected = interleaved ? g+sel[i]*groups : g*batch+sel[i];
                int actual = teranoc::remap_port(input, ports, batch,
                                                interleaved, cycle % batch);
                assert(actual == expected);
                assert(!seen[actual]);
                seen[actual] = true;
            }
            std::rotate(sel.begin(), sel.begin()+1, sel.end());
        }
    }
    std::cout << "PASS remapper: all ports, phases, and bundle modes\n";
}
