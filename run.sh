#!/bin/bash
export ENTER_POINT='/EMBED'
export CYJS='Signature_Graph_17041nodes_0.56_ERSC.cyjs'
export RURL='http://192.168.99.100:23239/custom/SigineDMOA'
export MONGOURI='mongodb://127.0.0.1:27017/'
python app.py
unset ENTER_POINT
unset CYJS
