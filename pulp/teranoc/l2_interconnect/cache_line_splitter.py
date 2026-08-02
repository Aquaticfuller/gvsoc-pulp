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

import gvsoc.systree


class CacheLineSplitter(gvsoc.systree.Component):
    """Split io_v1 requests so that no child crosses a cache-line boundary.

    Children are issued sequentially through ``output``. The component collapses
    synchronous, pending, and denied child completions back into the original
    parent request. A denied io_v1 child is retained until the downstream target
    grants that same request object; it is never reissued.
    """

    def __init__(self, parent: gvsoc.systree.Component, name: str, line_size: int):
        super().__init__(parent, name)

        if line_size <= 0 or line_size & (line_size - 1):
            raise ValueError('line_size must be a positive power of two')

        self.add_sources(['pulp/teranoc/l2_interconnect/cache_line_splitter.cpp'])
        self.add_property('line_size', line_size)

    def i_INPUT(self) -> gvsoc.systree.SlaveItf:
        return gvsoc.systree.SlaveItf(self, 'input', signature='io')

    def o_OUTPUT(self, itf: gvsoc.systree.SlaveItf):
        self.itf_bind('output', itf, signature='io')
