#
# Copyright (C) 2024 ETH Zurich and University of Bologna
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
# Discription: This file is the GVSoC configuration file for the TeraNoc System.
# Author: Yinrong Li (ETH Zurich) (yinrli@student.ethz.ch)
#         Yichao Zhang (ETH Zurich) (yiczhang@iis.ee.ethz.ch)

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.10-3.11

import memory.memory_v3 as memory
from vp.clock_domain import Clock_domain
from interco.router_v2 import Router, RouterConfig, RouterMapping, KIND_BANDWIDTH, KIND_BEAT
from pulp.stdout.stdout_v3_v2 import StdoutV2
from utils.loader.loader_v2 import ElfLoader
import gvsoc.systree as st
from gvrun.parameter import TargetParameter
from pulp.teranoc_v2.dma.teranoc_dma import TeranocDma
from elftools.elf.elffile import *
from utils.io_v2_single_req_width_adapter import IoV2SingleReqWidthAdapter
from utils.io_v2_beat_to_single_req_adapter import IoV2BeatToSingleReqAdapter
from pulp.teranoc_v2.teranoc_cluster import TeranocCluster
from pulp.teranoc_v2.ctrl_registers import CtrlRegisters
from pulp.teranoc_v2.l2_subsystem import L2_subsystem
from pulp.teranoc_v2.dpi_checker import TeranocDpiChecker
from pulp.teranoc_v2.l2_interconnect.l2_address_scrambler import L2AddressScrambler
from pulp.teranoc_v2.l2_interconnect.l2_noc import L2_noc
from pulp.teranoc_v2.arch import CONFIGS, DEFAULT_CONFIG


def _add_config_arg(parser):
    if parser is None:
        return

    for action in parser._actions:
        if '--config' in action.option_strings:
            return

    parser.add_argument('--config', dest='config', choices=list(CONFIGS.keys()), default=None,
        help='Select Teranoc arch profile ' '(legacy alias for system_config)')


def _get_option(options, *names):
    for opt in options or []:
        key, sep, val = opt.partition('=')
        if sep and key.strip() in names:
            return val.strip()
    return None


def _get_dpi_check_symbols(binary):
    if binary is None:
        return 0, 0

    check_count_addr = 0
    check_table_addr = 0

    try:
        with open(binary, 'rb') as file:
            elffile = ELFFile(file)
            for section_name in ('.symtab', '.dynsym'):
                symtab = elffile.get_section_by_name(section_name)
                if symtab is None:
                    continue
                for symbol in symtab.iter_symbols():
                    if symbol.name == 'mempool_dpi_check_count':
                        check_count_addr = symbol.entry['st_value']
                    elif symbol.name == 'mempool_dpi_checks':
                        check_table_addr = symbol.entry['st_value']
    except Exception:
        return 0, 0

    return check_count_addr, check_table_addr


class TeranocSystem(st.Component):

    def __init__(self, parent, name, parser, arch, binary=None):
        super().__init__(parent, name)

        # Convenience locals for the bits we use heavily below; everything
        # else is read straight from arch.
        axi_data_width = arch.axi_data_width
        nb_banks_per_group = arch.nb_banks_per_group
        l2_bank_size = arch.l2_bank_size
        dma_width = arch.dma_data_width
        dma_stripe_bytes = arch.l1_bank_width * nb_banks_per_group

        ################################################################
        ##########              Design Components             ##########
        ################################################################

        # TeraNoC cluster
        teranoc_cluster = TeranocCluster(self, 'teranoc_cluster', parser=parser, arch=arch)

        # Boot Rom. MemoryV3 defaults differ from v1, so state all
        # behavior-relevant properties explicitly.
        rom = memory.Memory(self, 'rom', config=memory.MemoryV3Config(
                size=0x1000, latency=0, atomics=False, truncate=False, stim_file=self.get_file_path(
                'pulp/chips/spatz/rom.bin')))

        # L2 NoC
        l2_noc = L2_noc(self, 'l2_noc', width=axi_data_width, nb_x_groups=arch.nb_x_groups,
            nb_y_groups=arch.nb_y_groups, ni_outstanding_reqs=32, router_input_queue_size=2)

        # L2 Memory
        l2_mem = L2_subsystem(self, 'l2_mem', nb_banks=arch.nb_l2_banks,
            bank_width=axi_data_width, size=arch.l2_size, port_bandwidth=axi_data_width)

        dpi_check_count, dpi_check_table = _get_dpi_check_symbols(binary)
        dpi_checker = TeranocDpiChecker(self, 'mempool_dpi_checker', nb_banks=arch.nb_l2_banks,
            bank_width=axi_data_width, interleave=arch.l2_axi_interleave,
            l2_base=0x80000000, l2_size=arch.l2_size, check_count_addr=dpi_check_count,
            check_table_addr=dpi_check_table)
        self.dpi_checker = dpi_checker

        # CSR
        csr = CtrlRegisters(self, 'ctrl_registers', wakeup_latency=5)

        # UART
        uart = StdoutV2(self, 'uart', max_cluster=1, max_core_per_cluster=1,
            user_set_core_id=0, user_set_cluster_id=0)

        # DMA. Keep v1's split/distribution stages and group ownership, while
        # using the stock public single-path IOv2 backend like the RTL.
        dma = TeranocDma(self, 'dma', loc_base=0x0, loc_size=arch.l1_total_bytes,
            stripe_bytes=dma_stripe_bytes,
            # The RTL package fallback is four, but every active TeraNoC
            # configuration overrides DMAS_PER_GROUP to one.
            # One worker slice is one legal RTL DMA burst. The transport beat
            # remains AXI-wide and the public backend streams the slice.
            burst_size=dma_stripe_bytes, nb_groups=arch.nb_groups, nb_dmas_per_group=1,
            axi_width=dma_width, transfer_queue_size=16, burst_queue_size=8)

        # Binary Loader
        loader = ElfLoader(self, 'loader', binary=binary, entry=0x80000000)
        self.loader = loader

        # L2 loader converter. Keep the dedicated single-request splitter here:
        # unlike the live-fabric interleaver, it preserves the loader request's
        # ownership and data pointer across every granule of a large ELF write.
        l2_loader_converter = IoV2SingleReqWidthAdapter(self, 'l2_loader_converter',
            width=axi_data_width * arch.l2_axi_interleave)

        # L2 loader address scrambler
        l2_loader_scrambler = L2AddressScrambler(self, 'l2_loader_scrambler',
            bypass=False, l2_base_addr=0x0, l2_size=arch.l2_size, nb_banks=arch.nb_l2_banks,
            bank_width=axi_data_width, interleave=arch.l2_axi_interleave)

        # Dummy Memory
        dummy_mem = memory.Memory(self, 'dummy_mem', config=memory.MemoryV3Config(
                size=arch.l1_total_bytes, latency=0, atomics=True, truncate=False))

        soc_ico = Router(self, 'soc_ico', config=RouterConfig(kind=KIND_BANDWIDTH))

        periph_ico = Router(self, 'periph_ico', config=RouterConfig(
                kind=KIND_BANDWIDTH, bandwidth=4))

        ext_ico = Router(self, 'ext_ico', config=RouterConfig(kind=KIND_BANDWIDTH, bandwidth=4))

        # Binary Loader Router. Bandwidth 0 = unshaped: the ELF load is a back
        # door with no hardware counterpart (the RTL testbench preloads L2
        # directly at t=0), so shaping it only inflates the pre-boot phase.
        loader_router = Router(self, 'loader_router', config=RouterConfig(
                kind=KIND_BANDWIDTH, bandwidth=0, latency=1))

        ################################################################
        ##########               Design Bindings              ##########
        ################################################################

        # Group axi port -> L2 NoC
        for i in range(0, arch.nb_x_groups):
            for j in range(0, arch.nb_y_groups):
                teranoc_cluster.o_AXI(i, j, 0, l2_noc.i_CLUSTER_WIDE_INPUT(i, j))

        # L2 NoC -> HBM and peripherals
        if arch.nb_x_groups == 2 and arch.nb_y_groups == 2:
            soc_demux = Router(self, 'soc_demux', config=RouterConfig(kind=KIND_BANDWIDTH,
                    bandwidth=axi_data_width, latency=4))
            soc_beat_adapter = IoV2BeatToSingleReqAdapter(self, 'soc_beat_adapter',
                beat_width=axi_data_width, max_read_bursts=32)

            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(0), x=0, y=1)  # HBM bank 0
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(1), x=0, y=2)  # HBM bank 1
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(2), x=3, y=2)  # HBM bank 2
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(3), x=3, y=1)  # HBM bank 3
            l2_noc.o_WIDE_BIND(
                soc_beat_adapter.i_INPUT(), x=1, y=0)  # soc
            soc_beat_adapter.o_OUTPUT(soc_demux.i_INPUT(0))

            l2_noc.o_MAP(base=0x80000000+l2_bank_size*0, size=l2_bank_size,
                x=0, y=1, name='hbm0', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*1, size=l2_bank_size,
                x=0, y=2, name='hbm1', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*2, size=l2_bank_size,
                x=3, y=2, name='hbm2', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*3, size=l2_bank_size,
                x=3, y=1, name='hbm3', rm_base=True)
            l2_noc.o_MAP(base=0x00000000, size=0x80000000, x=1, y=0, name='soc1', rm_base=False)
            l2_noc.o_MAP(base=0xA0000000, size=0x30000000, x=1, y=0, name='soc2', rm_base=False)

            soc_demux.o_MAP_DEFAULT(soc_ico.i_INPUT(), name='soc')

        elif arch.nb_x_groups == 4 and arch.nb_y_groups == 4:
            hbm5_soc_demux = Router(self, 'hbm5_soc_demux', config=RouterConfig(kind=KIND_BEAT,
                    width=axi_data_width, latency=4, max_pending_bursts_per_input=32))
            hbm5_soc_beat_adapter = IoV2BeatToSingleReqAdapter(self, 'hbm5_soc_beat_adapter',
                beat_width=axi_data_width, max_read_bursts=32)

            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(0), x=0, y=1)  # HBM bank 0
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(1), x=0, y=2)  # HBM bank 1
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(2), x=0, y=3)  # HBM bank 2
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(3), x=0, y=4)  # HBM bank 3
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(4), x=2, y=0)  # HBM bank 4
            l2_noc.o_WIDE_BIND(hbm5_soc_demux.i_INPUT(0),
                x=1, y=0)  # HBM bank 5 + soc
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(6), x=1, y=5)  # HBM bank 6
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(7), x=2, y=5)  # HBM bank 7
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(8), x=3, y=0)  # HBM bank 8
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(9), x=4, y=0)  # HBM bank 9
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(10), x=4, y=5)  # HBM bank 10
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(11), x=3, y=5)  # HBM bank 11
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(12), x=5, y=1)  # HBM bank 12
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(13), x=5, y=2)  # HBM bank 13
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(14), x=5, y=3)  # HBM bank 14
            l2_noc.o_WIDE_BIND(
                l2_mem.i_BANK_INPUT(15), x=5, y=4)  # HBM bank 15

            l2_noc.o_MAP(base=0x80000000+l2_bank_size*0, size=l2_bank_size,
                x=0, y=1, name='hbm0', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*1, size=l2_bank_size,
                x=0, y=2, name='hbm1', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*2, size=l2_bank_size,
                x=0, y=3, name='hbm2', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*3, size=l2_bank_size,
                x=0, y=4, name='hbm3', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*4, size=l2_bank_size,
                x=2, y=0, name='hbm4', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*5, size=l2_bank_size,
                x=1, y=0, name='hbm5', rm_base=False)
            l2_noc.o_MAP(base=0x00000000, size=0x80000000, x=1, y=0, name='soc1', rm_base=False)
            l2_noc.o_MAP(base=0xA0000000, size=0x30000000, x=1, y=0, name='soc2', rm_base=False)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*6, size=l2_bank_size,
                x=1, y=5, name='hbm6', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*7, size=l2_bank_size,
                x=2, y=5, name='hbm7', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*8, size=l2_bank_size,
                x=3, y=0, name='hbm8', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*9, size=l2_bank_size,
                x=4, y=0, name='hbm9', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*10, size=l2_bank_size,
                x=4, y=5, name='hbm10', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*11, size=l2_bank_size,
                x=3, y=5, name='hbm11', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*12, size=l2_bank_size,
                x=5, y=1, name='hbm12', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*13, size=l2_bank_size,
                x=5, y=2, name='hbm13', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*14, size=l2_bank_size,
                x=5, y=3, name='hbm14', rm_base=True)
            l2_noc.o_MAP(base=0x80000000+l2_bank_size*15, size=l2_bank_size,
                x=5, y=4, name='hbm15', rm_base=True)

            hbm5_soc_demux.o_MAP(l2_mem.i_BANK_INPUT(5), mapping=RouterMapping(name='hbm5',
                    base=0x80000000+l2_bank_size*5, size=l2_bank_size, remove_base=True))
            hbm5_soc_demux.o_MAP_DEFAULT(hbm5_soc_beat_adapter.i_INPUT(), name='soc')
            hbm5_soc_beat_adapter.o_OUTPUT(soc_ico.i_INPUT())

        else:
            raise ValueError('teranoc_v2 preserves the v1 L2 endpoint topology, which '
                'defines only 2x2 and 4x4 group meshes; '
                f'got {arch.nb_x_groups}x{arch.nb_y_groups}')

        # Peripheral interconnect
        soc_ico.o_MAP(periph_ico.i_INPUT(), mapping=RouterMapping(
                name='peripheral', base=0x40000000, size=0x20000, remove_base=False, latency=1))

        # External interconnect
        soc_ico.o_MAP_DEFAULT(ext_ico.i_INPUT(), name='external', latency=1)

        # Bootrom
        soc_ico.o_MAP(rom.i_INPUT(), mapping=RouterMapping(name='bootrom', base=0xa0000000,
                size=0x10000, remove_base=True, latency=1))

        # CSR
        periph_ico.o_MAP(csr.i_INPUT(), mapping=RouterMapping(name='csr', base=0x40000000,
                size=0x10000, remove_base=True, latency=1))
        csr.o_DPI_CHECK(st.SlaveItf(dpi_checker, 'input', signature='wire<int>'))
        for i in range(0, arch.nb_l2_banks):
            self.bind(dpi_checker, f'meminfo_{i}', l2_mem, f'meminfo_{i}')

        # DMA Ctrl
        periph_ico.o_MAP(dma.i_INPUT(), mapping=RouterMapping(name='dma_ctrl', base=0x40010000,
                size=0x10000, remove_base=True, latency=1))

        # UART
        ext_ico.o_MAP(uart.i_INPUT(), mapping=RouterMapping(name='uart', base=0xc0000000,
                size=0x100, remove_base=True, latency=1))

        # Loader router
        loader.o_START(teranoc_cluster.i_LOADER_START())
        loader.o_ENTRY(teranoc_cluster.i_LOADER_ENTRY())
        loader.o_OUT(loader_router.i_INPUT())
        loader_router.o_MAP(dummy_mem.i_INPUT(), mapping=RouterMapping(
                name='dummy', base=0x00000000, size=arch.l1_total_bytes, remove_base=True))
        loader_router.o_MAP(l2_loader_converter.i_INPUT(), mapping=RouterMapping(
                name='mem', base=0x80000000, size=arch.l2_size, remove_base=True))
        loader_router.o_MAP(rom.i_INPUT(), mapping=RouterMapping(name='rom', base=0xa0000000,
                size=0x1000, remove_base=True))
        loader_router.o_MAP(csr.i_INPUT(), mapping=RouterMapping(name='csr', base=0x40000000,
                size=0x10000, remove_base=True))
        l2_loader_converter.o_OUTPUT(l2_loader_scrambler.i_INPUT())
        l2_loader_scrambler.o_OUTPUT(l2_mem.i_LOADER())

        # Cluster Registers for synchronization barrier
        for i in range(0, arch.total_snitch):
            csr.o_BARRIER_ACK(teranoc_cluster.i_BARRIER_ACK(i))

        # L2 ro-cache configuration
        csr.o_ROCACHE_CFG(teranoc_cluster.i_ROCACHE_CFG())

        # DMA data. Each distributed worker has one public AXI read/write pair;
        # the owning group routes addresses back to L1 or to its external path.
        for i in range(arch.nb_groups):
            dma.o_AXI_READ(i, 0, teranoc_cluster.i_DMA_AXI_READ(i))
            dma.o_AXI_WRITE(i, 0, teranoc_cluster.i_DMA_AXI_WRITE(i))


class TeranocSoc(st.Component):
    # Override in subclasses to change the default profile.
    SYSTEM_CONFIG_DEFAULT = DEFAULT_CONFIG

    def __init__(self, parent, name, parser, options):
        super().__init__(parent, name, options=options)

        _add_config_arg(parser)
        [args, __] = parser.parse_known_args()

        config_name = self.SYSTEM_CONFIG_DEFAULT
        option_config = _get_option(options, 'system_config', 'config')
        if option_config is not None:
            config_name = option_config
        legacy_config = getattr(args, 'config', None)
        if legacy_config is not None:
            config_name = legacy_config

        TargetParameter(self, name='system_config', value=config_name,
            allowed_values=list(CONFIGS.keys()), cast=str, description='Teranoc arch profile')
        config_name = self.get_parameter('system_config')
        if config_name not in CONFIGS:
            raise ValueError(f"Invalid system_config '{config_name}'; "
                f"choose from {list(CONFIGS.keys())}")
        arch = CONFIGS[config_name]

        binary = getattr(args, 'binary', None)

        TargetParameter(self, name='binary', value=None,
            description='ELF binary to load and start', cast=str)

        clock = Clock_domain(self, 'clock', frequency=500000000)
        system = TeranocSystem(self, 'teranoc_system', parser, arch=arch, binary=binary)
        self.bind(clock, 'out', system, 'clock')
        self.loader = system.loader
        self.dpi_checker = system.dpi_checker
        self.register_binary_handler(self.handle_binary)

    @override
    def configure(self):
        binary = self.get_parameter('binary')
        if binary is not None:
            self.loader.set_binary(binary)
            dpi_check_count, dpi_check_table = (_get_dpi_check_symbols(binary))
            self.dpi_checker.set_symbols(dpi_check_count, dpi_check_table)

    def handle_binary(self, binary: str):
        self.set_parameter('binary', binary)
