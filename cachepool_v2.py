#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from pulp.cachepool_v2.cachepool_v2_system import CachepoolV2System
import gvsoc.runner as gvsoc

GAPY_TARGET = True

class Target(gvsoc.Target):

    def __init__(self, parser, options):
        super(Target, self).__init__(parser, options,
            model=CachepoolV2System, description="CachepoolV2System")
