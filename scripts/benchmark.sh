#!/bin/sh

P=`dirname $0`/..
test -d $P/outputs || mkdir $P/outputs
export OMP_NUM_THREADS

files=`ls $P/sample_nn_data/*.tst | sed 's/\.tst$//'`

for f in $files
do
    has_floats=`awk '$1 == "floats" { print $2 }' $f.tst.info`
    if [ $has_floats = yes ]
    then
        types='float double'
    else
        types='byte short int float double'
    fi

    for impl in simple unroll2 unroll4
    do
        sf='--scalar'
        for mode in sca vec
        do
            for t in $types
            do
                for bmegs in 0 4
                do
                    bs=`expr $bmegs '*' 1024 '*' 1024`
                    for nt in 1 2 3 4 5 6 7 8
                    do
                        output=$P/outputs/`basename $f`-$impl-$mode-b$bmegs-$t-$nt
                        test -f $output && continue
                        echo "--> $output"
                        OMP_NUM_THREADS=$nt
                        $P/$impl --runs=5 --type=$t $sf --blocksize=$bs $f | tee $output
                    done
                done
            done
            sf=
        done
    done
done
