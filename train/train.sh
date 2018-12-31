#!/usr/bin/env sh
/home/wfw/caffe-master/build/tools/caffe train --solver=/home/wfw/lipj/TMD/solver.prototxt --snapshot=/home/wfw/lipj/TMD/snapshot/TMD_renetxt__iter_100000.solverstate 2>&1   | tee /home/wfw/lipj/TMD/TMD_out.log  #--gpu 0 --weights=/home/wfw/lipj/TMD/snapshot/TMD_iter_100000.caffemodel
