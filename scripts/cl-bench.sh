#!/bin/sh

. ./paths.sh
ensure_path $cl_outputs
wd=$(pwd)

for file in $cl_inputs/*.tst
do
    f=${file%.tst}
    bf=${f#$cl_inputs/}
    output=$cl_outputs/$bf

    if [ -f $output ]
    then
        continue
    fi

    echo "--> $output"
    cd ../opencl
    ./clnn $f | tee $output
    cd $wd
done

