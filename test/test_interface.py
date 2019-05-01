#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

import os
import tempfile
from sos_notebook.test_utils import NotebookTest


class TestInterface(NotebookTest):

    def test_prompt_color(self, notebook):
        '''test color of input and output prompt'''
        idx = notebook.call(
            '''\
            cat('this is R')
            ''', kernel='F#')
        assert [220, 220, 218] == notebook.get_input_backgroundColor(idx)
        assert [220, 220, 218] == notebook.get_output_backgroundColor(idx)

    def test_cd(self, notebook):
        '''Support for change of directory with magic %cd'''
        output1 = notebook.check_output('getwd()', kernel='F#')
        notebook.call('%cd ..', kernel="SoS")
        output2 = notebook.check_output('getwd()', kernel='F#')
        assert len(output1) > len(output2)
        assert output1.strip("'").startswith(output2.strip("'"))
        #
        # cd to a specific directory
        tmpdir = os.path.join(tempfile.gettempdir(), 'somedir')
        os.makedirs(tmpdir, exist_ok=True)
        notebook.call(f'%cd {tmpdir}', kernel="SoS")
        output = notebook.check_output('cat(getwd())', kernel='F#')
        assert os.path.realpath(tmpdir) == os.path.realpath(output)

    def test_var_names(self, notebook):
        '''Test get/put variables with strange names'''
        # _a_1 => .a_1 in R
        notebook.call(
            '''\
            %put _a_1 --to R
            _a_1 = 5
            ''',
            kernel="SoS",
            expect_error=True)
        assert '5' == notebook.check_output('.a_1', kernel='R')
        # .a.1 => _a_1 in Python (SoS)
        notebook.call(
            '''\
            %put .a.1
            .a.1 = 500
            ''',
            kernel='F#',
            expect_error=True)
        assert '500' == notebook.check_output('_a_1', kernel='SoS')

    def test_auto_vars(self, notebook):
        '''Test automatic exchange of variables with names starting with sos'''
        notebook.call('sosInSoS=123', kernel="SoS")
        assert '123' == notebook.check_output('sosInSoS', kernel='R')

        notebook.call('sosInR <- 12345', kernel='F#')
        assert '12345' == notebook.check_output('sosInR', kernel='SoS')

    def test_preview(self, notebook):
        '''Test support for %preview'''
        output = notebook.check_output(
            '''\
            %preview -n var
            var = seq(1, 1000)
            ''',
            kernel='F#')
        # in a normal var output, 100 would be printed. The preview version would show
        # type and some of the items in the format of
        #   int [1:1000] 1 2 3 4 5 6 7 8 9 10 ...
        assert 'int' in output and '3' in output and '9' in output and '111' not in output
        #
        # return 'Unknown variable' for unknown variable
        assert 'Unknown variable' in notebook.check_output(
            '%preview -n unknown_var', kernel='F#')
        #
        # return 'Unknown variable for expression
        assert 'Unknown variable' in notebook.check_output(
            '%preview -n var[1]', kernel='F#')

    def test_sessioninfo(self, notebook):
        '''test support for %sessioninfo'''
        notebook.call("cat('this is R')", kernel='F#')
        assert 'R version' in notebook.check_output(
            '%sessioninfo', kernel="SoS")
