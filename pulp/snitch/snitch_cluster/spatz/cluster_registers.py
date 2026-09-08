#
# Copyright (C) 2020 GreenWaves Technologies, SAS, ETH Zurich and University of Bologna
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

import gvsoc.systree
import regmap.regmap
import regmap.regmap_hjson
import regmap.regmap_c_header

class ClusterRegisters(gvsoc.systree.Component):

    def __init__(self, parent, name, boot_addr=0, nb_cores=1, binary=None, cachepool=False, nb_flush=0,
                 nb_config=0, cachepool_map='legacy'):
        super(ClusterRegisters, self).__init__(parent, name)

        self.add_sources(['pulp/snitch/snitch_cluster/spatz/cluster_registers.cpp'])

        self.add_properties({
            'boot_addr': boot_addr,
            'nb_cores': nb_cores,
            # CachePool mode: also accept the CachePool peripheral register block (L1D-config 0x28..0x4c
            # as RW scratch, FLUSH_STATUS 0x3c reads 0, EOC 0x24 -> quit) so the unmodified snrt CachePool
            # binaries don't fault on "invalid register". Default off -> the spatz regmap is unchanged.
            'cachepool': cachepool,
            # Which generation of the CachePool peripheral map to answer. 'legacy' = HW_BARRIER 0x10 /
            # BOOT_CONTROL 0x20 / EOC 0x24, what every ELF in software/build/CachePoolTests uses.
            # 'multi_scalar' = the dev/multi-scalar map, where SPATZ_LOCK_ACQUIRE/RELEASE were inserted
            # at 0x4/0x8 and pushed HW_BARRIER to 0x00, BOOT_CONTROL to 0x18 and EOC to 0x1c. Picking
            # the wrong one fails SILENTLY: the barrier stops blocking and the EOC write goes nowhere.
            'cachepool_map': cachepool_map,
            # F1: number of insitu-cache cells the COMMIT flush fans out to (0 = scratch fallback,
            # FLUSH_STATUS pinned 0). The cluster wires one flush_out_N per cell when set.
            'nb_flush': nb_flush,
            # E3: number of partition-config endpoints (xbar + core cells + remote xbars) the
            # partition-commit config broadcast fans out to. 0 = no config path (non-structural builds).
            'nb_config': nb_config,
        })

    def o_FLUSH(self, port: int, itf: gvsoc.systree.SlaveItf):
        """Flush fan-out master ``port`` → one insitu-cache cell's flush slave port."""
        self.itf_bind(f'flush_out_{port}', itf, signature='io')

    def o_CONFIG(self, itf: gvsoc.systree.SlaveItf):
        """E3 partition-config master → the config broadcast shim (one partition-commit write fans
        out to every xbar / core cell / remote xbar)."""
        self.itf_bind('config_out', itf, signature='io')

    def gen(self, builddir, installdir):
        comp_path = 'pulp/snitch/snitch_cluster/spatz'
        regmap_path = f'{comp_path}/spatz_cluster_peripheral_reg.hjson'
        regmap_instance = regmap.regmap.Regmap('cluster_periph')
        regmap.regmap_hjson.import_hjson(regmap_instance, self.get_file_path(regmap_path))
        regmap.regmap_c_header.dump_to_header(regmap=regmap_instance, name='cluster_periph',
            header_path=f'{builddir}/{comp_path}/cluster_periph', headers=['regfields', 'gvsoc'])

    def o_EXTERNAL_IRQ(self, core: int, itf: gvsoc.systree.SlaveItf):
        self.itf_bind(f'external_irq_{core}', itf, signature='wire<bool>')

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature='io')

    def i_CORE_INPUT(self, core) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'input_{core}', signature='io')

    def i_BARRIER_ACK(self, core: int) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, f'barrier_req_{core}', signature='wire<bool>')

    def gen_gui(self, parent_signal):
        return gvsoc.gui.Signal(self, parent_signal, name=self.name, is_group=True, groups=["regmap"])
