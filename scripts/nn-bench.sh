#!/bin/sh

. ./paths.sh
ensure_path $nn_outputs
export OMP_NUM_THREADS
l3block=$((2 * 1024 * 1024))
l2block=$((128 * 1024))

#
# Parameters: {impl} {mode} {type} {basename}
#
unblocked()
{
    output=$nn_outputs/$4-$1-$2-$3-b0-$OMP_NUM_THREADS
    test -f $output && return
    s=
    test $2 = sca && s=--scalar
    echo "--> ${output#$nn_outputs/}"
    ../$1 --runs=3 $s --type=$3 $nn_inputs/$4 | tee $output
}

single_blocking()
{
    output=$nn_outputs/$4-$1-$2-$3-b1-$OMP_NUM_THREADS
    test -f $output && return
    s=
    test $2 = sca && s=--scalar
    echo "--> ${output#$nn_outputs/}"
    ../$1 --runs=3 $s --type=$3 --blocksize=$l3block $nn_inputs/$4 | tee $output
}

double_blocking()
{
    output=$nn_outputs/$4-$1-$2-$3-b2-$OMP_NUM_THREADS
    test -f $output && return
    s=
    test $2 = sca && s=--scalar
    echo "--> ${output#$nn_outputs/}"
    ../$1 --runs=3 $s --type=$3 --blocksize=$l2block --superblock=$l3block \
          $nn_inputs/$4 | tee $output
}


for file in $nn_inputs/*.tst
do
    f=${file%.tst}
    bf=${f#$nn_inputs/}
    has_floats=$(awk '$1 == "floats" { print $2 }' $f.tst.info)

    if [ $has_floats = yes ]
    then
        types='float double'
        tk=float
    else
        types='byte short int float double'
        tk=byte
    fi

    if [ ${1:-basic} = full ]
    then
        #
        # Full benchmark
        #
        for impl in simple unroll2 unroll4
        do
            for mode in sca vec
            do
                for t in $types
                do
                    for OMP_NUM_THREADS in 1 2 3 4 5 6 7 8
                    do
                        unblocked $impl $mode $t $bf
                        single_blocking $impl $mode $t $bf
                        double_blocking $impl $mode $t $bf
                    done
                done
            done
        done
    else
        #
        # Basic benchmark
        #
        OMP_NUM_THREADS=4
        double_blocking simple sca $tk $bf
        OMP_NUM_THREADS=1
        for t in $types
        do
            unblocked simple sca $t $bf
            for impl in simple unroll2 unroll4
            do
                for mode in sca vec
                do
                    single_blocking $impl $mode $t $bf
                    double_blocking $impl $mode $t $bf
                done
            done
        done
    fi
done

