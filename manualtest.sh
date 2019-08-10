#!/bin/bash
echo "clearing debug log, killing mono processes (including ifsharp.exe), and launching jupyter notebook"
rm ~/.sos/sos_debug.log 
killall mono
SOS_DEBUG=ALL; export SOS_DEBUG
echo "logging to ~/.sos/sos_debug.log"
jupyter notebook
