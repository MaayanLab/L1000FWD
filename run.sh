#!/bin/bash
export ENTER_POINT='/EMBED'
export CYJS='Signature_Graph_17041nodes_0.56_ERSC.cyjs'
# export RURL='http://146.203.54.140:31047/custom/SigineDMOA'
export RURL='http://146.203.54.239:31722/custom/SigineDMOA'
export MONGOURI='mongodb://146.203.54.131:27017/L1000FWD'
# export MONGOURI='mongodb://127.0.0.1:27017/L1000FWD'
export MYSQLURI='mysql://root:@127.0.0.1:3306/L1000FWD?charset=utf8'
python app.py
unset ENTER_POINT
unset CYJS
