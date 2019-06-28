#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

import os
import pandas as pd
import numpy
import re
from sos.utils import short_repr, env
from IPython.core.error import UsageError

from ._version import __version__

def homogeneous_type(seq):
    iseq = iter(seq)
    first_type = type(next(iseq))
    if first_type in (int, float):
        return True if all(isinstance(x, (int, float)) for x in iseq) else False
    else:
        return True if all(isinstance(x, first_type) for x in iseq) else False



#
#  support for %get
#
#  Converting a Python object to an Fsharp expression that will be executed
#  by the ifsharp kernel.
#
#


def _Fsharp_repr(obj, processed=None):
    if isinstance(obj, bool):
        return 'true' if obj else 'false'
    elif isinstance(obj, (int, str)):
        return repr(obj)
    elif isinstance(obj, float):
        if numpy.isnan(obj):
            return 'NaN'
        elif numpy.isinf(obj):
            return 'Infinity'
        else:
            return repr(obj)
    elif isinstance(obj, Sequence):
        if len(obj) == 0:
            return 'Seq.empty |> ResizeArray'
        # if the data is of homogeneous type, let us use typed ResizeArray
        # otherwise use untyped ResizeArray
        # this can be confusing but list can be difficult to handle
        if homogeneous_type(obj):
            return '[' + ';'.join(_Fsharp_repr(x) for x in obj) + '] |> ResizeArray'
        else:
            return '[' + ':> obj;'.join(_Fsharp_repr(x) for x in obj) + '] |> ResizeArray'
    elif obj is None:
        return 'null'
    elif isinstance(obj, dict):
        if processed:
            if id(obj) in processed:
                return 'null'
        else:
            processed = set()
        processed.add(id(obj))
        return '[' + ';'.join(
            '({},{})'.format(str(x), _Fsharp_repr(y, processed))
            for x, y in obj.items()) + '] |> Map.ofList |> System.Collections.Generic.Dictionary'
    elif isinstance(obj, set):
        return '[' + ';'.join(_Fsharp_repr(x) for x in obj) + '] |> System.Collections.Generic.HashSet'
    elif isinstance(
            obj, (numpy.intc, numpy.intp, numpy.int8, numpy.int16, numpy.int32,
                  numpy.int64, numpy.uint8, numpy.uint16, numpy.uint32,
                  numpy.uint64, numpy.float16, numpy.float32, numpy.float64)):
        return repr(obj)
    elif isinstance(obj, numpy.matrixlib.defmatrix.matrix):
        # try:
        #     import feather
        # except ImportError:
            # raise UsageError(
            #     'Getting numpy matrix is not implemented.'
            # )
        return repr('Unsupported datatype {}'.format(short_repr(obj)))
    elif isinstance(obj, numpy.ndarray):
        if obj.ndim == 1:
            return '[|' + ';'.join(_Fsharp_repr(x) for x in obj) + '|]'
        else:
            return repr('Unsupported datatype {}'.format(short_repr(obj)))
            # return 'array(' + 'c(' + ','.join(
            #     repr(x)
            #     for x in obj.swapaxes(obj.ndim - 2, obj.ndim - 1).flatten(
            #         order='C')) + ')' + ', dim=(' + 'rev(c' + repr(
            #             obj.swapaxes(obj.ndim - 2, obj.ndim - 1).shape) + ')))'
    elif isinstance(obj, pandas.DataFrame):
        # try:
        #     import feather
        # except ImportError:
            # raise UsageError(
            #     'Getting pandas DataFrame is not implemented.'
            # )
        return repr('Unsupported datatype {}'.format(short_repr(obj)))
        # feather_tmp_ = tempfile.NamedTemporaryFile(
        #     suffix='.feather', delete=False).name
        # try:
        #     data = obj.copy()
        #     # if the dataframe has index, it would not be transferred due to limitations
        #     # of feather. We will have to do something to save the index separately and
        #     # recreate it. (#397)
        #     if isinstance(data.index, pandas.Index):
        #         df_index = list(data.index)
        #     elif not isinstance(data.index, pandas.RangeIndex):
        #         # we should give a warning here
        #         df_index = None
        #     feather.write_dataframe(data, feather_tmp_)
        # except Exception:
        #     # if data cannot be written, we try to manipulate data
        #     # frame to have consistent types and try again
        #     for c in data.columns:
        #         if not homogeneous_type(data[c]):
        #             data[c] = [str(x) for x in data[c]]
        #     feather.write_dataframe(data, feather_tmp_)
        #     # use {!r} for path because the string might contain c:\ which needs to be
        #     # double quoted.
        # return '..read.feather({!r}, index={})'.format(feather_tmp_,
        #                                                _Fsharp_repr(df_index))
    elif isinstance(obj, pandas.Series):
        dat = list(obj.values)
        ind = list(obj.index.values)
        return 'List.zip ' + '[' + ';'.join(
            _Fsharp_repr(x) for x in dat) + '] ' + '[' + ';'.join(
                _Fsharp_repr(y) for y in ind) + ']'
    else:
        return repr('Unsupported datatype {}'.format(short_repr(obj)))


class sos_fsharp:
    supported_kernels = {'F#': ['ifsharp']}
    background_color = '#5DBCD2'
    options = {
        'variable_pattern': r'^\s*let\s+([_A-Za-z\$@`\?][_A-Za-z0-9\$@`\?]+)\s*(=).*$',
        'assignment_pattern': r'^\s*([_A-Za-z\$@`\?][_A-Za-z0-9\$@`\?]+)\s*(<-).*$',
    }
    cd_command = "System.Environment.CurrentDirectory <- @\"{dir}\""
    __version__ = __version__

    def __init__(self, sos_kernel, kernel_name='ifsharp'):
        self.sos_kernel = sos_kernel
        self.kernel_name = kernel_name
        self.init_statements = ''

    def get_vars(self, names):
        for name in names:
            # https://stackoverflow.com/questions/41274652/what-characters-are-allowed-in-f-identifiers-module-type-and-member-names
            if name.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                self.sos_kernel.warn(
                    f'Variable {name} is passed from SoS to kernel {self.kernel_name} as {"?" + name[1:]}'
                )
                newname = '?' + name[1:]
            else:
                newname = name
            fsharp_repr = _Fsharp_repr(env.sos_dict[name])
            env.log_to_file('VARIABLE', fsharp_repr)
            self.sos_kernel.run_cell(
                f'let {newname} = {fsharp_repr}',
                True,
                False,
                on_error=f'Failed to get variable {name} to F#')

    def put_vars(self, items, to_kernel=None):
        # first let us get all variables with names starting with sos
        response = self.sos_kernel.get_response(
            'cat(..py.repr(System.Reflection.Assembly.GetExecutingAssembly().GetTypes() |> Seq.collect( fun t -> t.GetProperties(System.Reflection.BindingFlags.Static ||| System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Public) |> Seq.map(fun p -> p.Name) ) |> Seq.toArray))', ('stream',), name=('stdout',))[0][1]
        all_vars = eval(response['text'])
        all_vars = [all_vars] if isinstance(all_vars, str) else all_vars

        items += [x for x in all_vars if x.startswith('sos')]

        for item in items:
            if '.' in item:
                self.sos_kernel.warn(
                    f'Variable {item} is put to SoS as {item.replace(".", "_")}'
                )

        if not items:
            return {}

        py_repr = f'cat(..py.repr(list({",".join("{0}={0}".format(x) for x in items)})))'
        response = self.sos_kernel.get_response(
            py_repr, ('stream',), name=('stdout',))[0][1]
        expr = response['text']

        if to_kernel in ('Python2', 'Python3'):
            # directly to python3
            return '{}\n{}\nglobals().update({})'.format(
                'from feather import read_dataframe\n'
                if 'read_dataframe' in expr else '',
                'import numpy' if 'numpy' in expr else '', expr)
        # to sos or any other kernel
        else:
            # TODO NOT SURE HOW TO HANDLE THIS
            try:
                if 'read_dataframe' in expr:
                    # imported to be used by eval
                    from feather import read_dataframe
                    # suppress flakes warning
                    assert read_dataframe
                # evaluate as raw string to correctly handle \\ etc
                return eval(expr)
            except Exception as e:
                self.sos_kernel.warn(f'Failed to evaluate {expr!r}: {e}')
                return None

    def preview(self, item):
        # return '', f'Unknown variable {item}'
        try:
            return "", self.sos_kernel.get_response(
                f'..sos.preview("{item}")', ('stream',),
                name=('stdout',))[0][1]['text']
        except Exception as e:
            env.log_to_file('VARIABLE', f'Preview of {item} failed: {e}')
            return None

    def sessioninfo(self):
        response = self.sos_kernel.get_response(
            'match System.AppDomain.CurrentDomain.GetAssemblies() |> Seq.map( fun a -> a.GetName()) |> Seq.tryFind( fun name -> name.Name = \"FSharp.Core\") with |Some(x) -> x.ToString()| None -> \"No session information is available\"',
            ('stream',),
            name=('stdout',))[0]
        return response[1]['text']

        