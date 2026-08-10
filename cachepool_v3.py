#
# Copyright (C) 2026 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# CachePool v3 target: v1's calibrated structural InSitu cache inside v2's multi-group shell.
# Plan + phase gates: prompt/cachepool_v3_implementation_plan_2026-08-10.md

from pulp.cachepool_v3.cachepool_v3_system import CachepoolV3System
import gvsoc.runner as gvsoc

GAPY_TARGET = True


class Target(gvsoc.Target):

    def __init__(self, parser, options):
        super(Target, self).__init__(parser, options,
            model=CachepoolV3System, description="CachepoolV3System")
