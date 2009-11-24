#!/bin/sh

test -d results || mkdir results
export OMP_NUM_THREADS=8

files=`ls ../sample_nn_data/*.tst | sed 's/\.tst$//'`
bs=$((2 * 1024 * 1024))
output=precisions.txt
ram_dir='/dev/shm'
svm_train='../libsvm-2.9/svm-train'
svm_predict='../libsvm-2.9/svm-predict'

#
# Obtain the results
#
for f in $files
do
    bf=`basename $f`

    #
    # Strip the original result
    #
    test -f results/$bf.orig || awk '{ print $1 }' $f.tst >results/$bf.orig

    #
    # Execute the appropriate NN versions
    #
    has_floats=`awk '$1 == "floats" { print $2 }' $f.tst.info`
    if [ $has_floats = yes ]
    then
        types='float double'
    else
        types='byte short int float double'
    fi
    for t in $types
    do
        test -f results/$bf.nn-$t && continue
        ../unroll4 --type=$t --blocksize=$bs --output=results/$bf.nn-$t $f
    done

    #
    # Execute the LibSVM version
    #
    lf=../sample_svm_data/${bf%.scale}.scale
    rf=$ram_dir/${bf%.scale}.scale
    if [ ! -f results/$bf.libsvm ]
    then
        cp $lf.trn $rf.trn
        cp $lf.tst $rf.tst
        set - `cat $lf.params`
        c=$1
        g=$2
        $svm_train -c $c -g $g $rf.trn $rf.model
        $svm_predict $rf.tst $rf.model results/$bf.libsvm
        rm -f $rf.*
    fi
done


#
# Compare the results
#
printf "%-15s  %6s  %6s  %6s  %6s  %6s  %6s\n" 'Data file' byte short int float double SVM >$output
for f in $files
do
    bf=`basename $f`
    printf "%-15s" ${bf%.scale}
    for r in nn-byte nn-short nn-int nn-float nn-double libsvm
    do
        if [ -f results/$bf.$r ]
        then
            printf "  %6.2f" `./compare.py results/$bf.orig results/$bf.$r`
        else
            printf "  %6s" -
        fi
    done
    echo
done >>$output

