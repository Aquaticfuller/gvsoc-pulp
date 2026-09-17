#!/usr/bin/env python3
"""Compile the production shim selector and check both RTL address geometries."""
from pathlib import Path
import subprocess
import tempfile
p=Path(__file__).resolve().parents[2]/'pulp/teranoc_spatz/l1_interconnect/teranoc_l1_shim.cpp'
s=p.read_text(); start=s.index('int TeranocL1Shim::select('); end=s.index('\nvp::IoReqStatus TeranocL1Shim::input_req',start)
selector=s[start:end].replace('TeranocL1Shim::','Fixture::')
source=r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
enum {OUT_LOCAL, OUT_REMOTE, OUT_SOC};
struct Fixture {
 uint64_t l1_size,control_base,control_start,control_size,tile_id,tile_mask;
 int tile_shift,nb_tiles_per_group;
 bool control_enable;
 int select(uint64_t address) const;
};
'''+selector+r'''
int main() {
 unsigned checks=0;
 for (unsigned cores: {256u,1024u}) {
  uint64_t stride=cores*64, top=256*stride, start=0x20000000+240*stride;
  for(unsigned src=0;src<cores;++src) {
   Fixture f{top,0x20000000,start,16*stride,src,cores-1,6,16,true};
   for(unsigned dst: {src, src^1u, src^16u, cores-1}) {
    for(unsigned word=240;word<256;++word) for(unsigned bank=0;bank<16;++bank) {
     uint64_t data=word*stride+dst*64+bank*4;
     assert(f.select(data)==(src==dst?OUT_LOCAL:OUT_REMOTE));
     assert(f.select(0x20000000+data)==(src/16==dst/16?OUT_REMOTE:OUT_SOC));
     checks+=2;
    }
   }
   for(uint64_t address: {top,start-1,start+16*stride,uint64_t(0x40010000),uint64_t(0x80000000)}) {
    assert(f.select(address)==OUT_SOC);++checks;
   }
   f.control_enable=false;
   assert(f.select(start+src*64)==OUT_SOC);++checks;
   // Legacy selection is explicit and preserves the former own-tile bypass.
   f.control_enable=true;f.control_base=0;f.control_start=240*stride;
   assert(f.select(f.control_start+src*64)==OUT_LOCAL);++checks;
  }
 }
 printf("%u production decoder checks passed\n",checks);
}
'''
source='#include <initializer_list>\n'+source
with tempfile.TemporaryDirectory(prefix='gvsoc-control-decode-') as d:
 c=Path(d)/'test.cpp';out=Path(d)/'test';c.write_text(source)
 subprocess.run(['g++-14.2.0','-std=c++17','-O2',str(c),'-o',str(out)],check=True)
 subprocess.run([str(out)],check=True)
