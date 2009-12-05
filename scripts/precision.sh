#!/bin/sh

. ./paths.sh
ensure_path $plots
output=$plots/table-precisions.txt

#
# AWK scripts
#
HAS_FLOATS='$1 == "floats" { print $2 }'
NN_PREC='/^Classification/ { printf $3 }'
SVM_PREC='/^Accuracy/ { printf substr($3, 1, length($3) - 1) }'


printf "%-10s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s\n" 'Data file' SVM \
       LINEAR 1-NN 3-NN 5-NN 7-NN 11-NN 13-NN 17-NN 19-NN 23-NN >$output

for file in $nn_inputs/*.tst
do
    f=${file%.tst}
    bf=${f#$nn_inputs/}
    has_floats=$(awk "$HAS_FLOATS" $f.tst.info)
    if [ $has_floats = yes ]
    then
        t=float
    else
        t=byte
    fi

    printf '%-10s' ${bf%.scale}
    if [ -f $svm_outputs/${bf%.scale}.scale ]
    then
        printf ' %6.2f' $(awk "$SVM_PREC" $svm_outputs/${bf%.scale}.scale)
    else
        printf ' %6s' -
    fi
    if [ -f $lin_outputs/$bf ]
    then
        printf ' %6.2f' $(awk "$SVM_PREC" $lin_outputs/$bf)
    else
        printf ' %6s' -
    fi
    input=$nn_outputs/$bf-simple-sca-$t-b2-4
    if [ -f $input ]
    then
        printf ' %6.2f' $(awk "$NN_PREC" $input)
    else
        printf ' %6s' -
    fi
    for k in 3 5 7 11 13 17 19 23
    do
        input=$knn_outputs/$bf-$t-$k-4
        if [ -f $input ]
        then
            printf ' %6.2f' $(awk "$NN_PREC" $input)
        else
            printf ' %6s' -
        fi
    done
    echo
done >>$output

