#!/bin/sh

runs=5
wd=`pwd`
p=
if [ `basename $wd` = scripts ]
then
    p=../
fi

test -d ${p}outputs || mkdir ${p}outputs

files=`ls ${p}sample_nn_data/*.tst | sed 's/\.tst$//'`

for f in $files
do
    for impl in simple unroll2 unroll4
    do
        sf='--scalar'
        for mode in sca vec
        do
            for type in byte short int float double
            do
                for bmegs in 0 4
                do
                    output=${p}outputs/`basename $f`-$impl-$mode-b$bmegs-$type
                    bs=`expr $bmegs '*' 1024 '*' 1024`
                    ./$p$impl --runs=$runs --type=$type $sf --blocksize=$bs $f | tee $output
                done
            done
            sf=
        done
    done
done

