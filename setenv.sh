#!/bin/bash
#source this script; e.g. source setenv.sh, to run tests in Chrome browser with display
JUPYTER_TEST_BROWSER=live; export JUPYTER_TEST_BROWSER
echo $JUPYTER_TEST_BROWSER
SOS_DEBUG=ALL; export SOS_DEBUG
echo "logging to ~/.sos/sos_debug.log"
