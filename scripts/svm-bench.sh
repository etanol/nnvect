#!/bin/sh

P=`dirname $0`/..
test -d $P/svm-outputs || mkdir $P/svm-outputs

svm_train="$P/libsvm-2.9/svm-train"
svm_predict="$P/libsvm-2.9/svm-predict"
ram_dir='/dev/shm'

files=`ls $P/sample_svm_data/*.tst | sed 's/\.tst$//'`


for f in $files
do
    test -f $f.params || continue
    set - `cat $f.params`
    c=$1
    g=$2

    bf=`basename $f`
    rf=$ram_dir/$bf
    output=$P/svm-outputs/$bf
    test -f $output && continue

    cp $f.trn $rf.trn
    cp $f.tst $rf.tst

    echo "--> $bf training"
    echo '=== TRAINING ===' > $output
    $svm_train -c $c -g $g $rf.trn $rf.model 2>/dev/null >> $output

    echo "--> $bf prediction"
    echo '=== PREDICTION ===' >> $output
    $svm_predict $rf.tst $rf.model /dev/null 2>/dev/null >> $output

    rm -f $rf.*
done

