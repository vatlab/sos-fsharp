#!/bin/bash
echo "clearing debug log, killing mono processes (including ifsharp.exe), and launching jupyter notebook"
rm ~/.sos/sos_debug.log 
killall mono
jupyter notebook
