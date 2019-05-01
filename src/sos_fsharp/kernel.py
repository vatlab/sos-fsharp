#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

import os
import pandas as pd
from sos.utils import env

from ._version import __version__


class sos_fsharp:
    supported_kernels = {'F#': ['ifsharp']}
    background_color = '#5DBCD2'
    options = {}
    cd_command = "cd {dir}"
    __version__ = __version__

    def __init__(self, sos_kernel, kernel_name='scala'):
        self.sos_kernel = sos_kernel
        self.kernel_name = kernel_name
        self.init_statements = ''

    def get_vars(self, names):
        pass

    def put_vars(self, items, to_kernel=None):
        return {}

    def preview(self, item):
        return '', f'Unknown variable {item}'

    def sessioninfo(self):
        return ''
