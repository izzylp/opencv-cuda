#!/bin/bash

filename=$1

compile=`head -1 $filename | sed 's/\/\* Compile and run: //'`

echo $compile 

eval $compile

extension="${filename##*.}"
notextension="${filename%.*}"

#nvprof ./$notextension $2 $3
./$notextension $2 $3
