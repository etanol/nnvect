#!/bin/sh

. ./paths.sh
ensure_path $lin_outputs
liblinear='../liblinear-1.5'
ram_dir='/dev/shm'

for file in $lin_inputs/*.tst
do
    f=${file%.tst}
    bf=${f#$lin_inputs/}
    rf=$ram_dir/$bf
    output=$lin_outputs/$bf

    if [ -f $output ]
    then
        continue
    fi

    cp $f.trn $rf.trn
    cp $f.tst $rf.tst

    echo "--> $bf training"
    echo '=== TRAINING ===' > $output
    $liblinear/train $rf.trn $rf.model 2>/dev/null >> $output

    echo "--> $bf prediction"
    echo '=== PREDICTION ===' >> $output
    $liblinear/predict $rf.tst $rf.model /dev/null 2>/dev/null >> $output

    rm -f $rf.trn $rf.tst
done

