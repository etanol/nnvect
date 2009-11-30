#!/bin/sh

. ./paths.sh
ensure_path $knn_outputs
export OMP_NUM_THREADS=4
l3block=$((2 * 1024 * 1024))
l2block=$((128 * 1024))

for file in $knn_inputs/*.tst
do
    f=${file%.tst}
    bf=${f#$knn_inputs/}
    has_floats=$(awk '$1 == "floats" { print $2 }' $f.tst.info)

    if [ $has_floats = yes ]
    then
        t=float
    else
        t=byte
    fi

    for k in 3 5 7 11 13 17 19 23
    do
        output=$knn_outputs/$bf-$t-$k-$OMP_NUM_THREADS
        test -f $output && continue
        echo "--> ${output#$knn_outputs/}"
        ../simple --runs=3 --blocksize=$l2block --superblock=$l3block \
                  --type=$t --neighbours=$k $f | tee $output
    done
done

