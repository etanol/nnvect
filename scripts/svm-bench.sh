#!/bin/sh

. ./paths.sh
ensure_path $svm_outputs
libsvm='../libsvm-2.9'
ram_dir='/dev/shm'

for file in $svm_inputs/*.tst
do
    f=${file%.tst}
    c=
    g=
    if [ -f $f.params ]
    then
        set - $(cat $f.params)
        c=$1
        g=$2
    fi
    bf=${f#$svm_inputs/}
    rf=$ram_dir/$bf
    output=$svm_outputs/$bf

    if [ -f $output ]
    then
        continue
    fi

    cp $f.trn $rf.trn
    cp $f.tst $rf.tst

    echo "--> $bf training"
    echo '=== TRAINING ===' > $output
    $libsvm/svm-train ${c:+-c} $c ${g:+-g} $g $rf.trn $rf.model 2>/dev/null >> $output

    echo "--> $bf prediction"
    echo '=== PREDICTION ===' >> $output
    $libsvm/svm-predict $rf.tst $rf.model /dev/null 2>/dev/null >> $output

    rm -f $rf.*
done

