#!/bin/sh

P=`dirname $0`/..
test -d $P/svm-outputs || mkdir $P/svm-outputs

svm_train="$P/libsvm-2.9/svm-train"
svm_predict="$P/libsvm-2.9/svm-predict"

files=`ls $P/sample_svm_data/*.tst | sed 's/\.tst$//'`


for f in $files
do
    test -f $f.params || continue
    set - `cat $f.params`
    c=$1
    g=$2

    output=$P/svm-outputs/`basename $f`
    test -f $output && continue

    echo "--> $f training warm-up"
    $svm_train -c $c -g $g $f.trn $f.model >/dev/null 2>&1
    echo "--> $f training real"
    echo '=== TRAINING ===' > $output
    $svm_train -c $c -g $g $f.trn $f.model 2>/dev/null >> $output

    echo "--> $f prediction warm-up"
    $svm_predict $f.tst $f.model /dev/null >/dev/null 2>&1
    echo "--> $f prediction real"
    echo '=== PREDICTION ===' >> $output
    $svm_predict $f.tst $f.model /dev/null 2>/dev/null >> $output
done

