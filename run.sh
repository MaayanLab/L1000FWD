#!/bin/bash
export ENTER_POINT='/embed'
export CYJS='Signature_Graph_17041nodes_0.56_ERSC.cyjs'
export RURL='http://192.168.99.100:23239/custom/SigineDMOA'
python app.py
unset ENTER_POINT
unset CYJS
