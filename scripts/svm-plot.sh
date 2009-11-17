#!/bin/sh

P=`dirname $0`/..
for t in byte short int float double
do
    test -d $P/svm-plots/$t || mkdir -p $P/svm-plots/$t
done

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

    title=`basename $f`
    printf '=> %-16s  ...  ' $title


    for t in $types
    do
        data=$P/svm-plots/$t/table-$title.txt

        if :
        then
            awk '/^TRAINING TIME:/ { train = $3 }
                 /^PREDICT TIME:/  { predict = $3 }
                 END { printf "LibSVM  %f  %f\n", predict, train }' \
                 $P/svm-outputs/${title%.scale}.scale

            awk 'BEGIN { printf "NN-worst  " }
                 $2 == "Minimum" { printf "%f  -\n", $4 }' \
                 $P/outputs/$title-simple-sca-$t-1

            awk 'BEGIN { printf "NN-best  " }
                 $2 == "Minimum" { printf "%f  -\n", $4 }' \
                 $P/outputs/$title-unroll4-vec-$t-1
        fi >$data

        gnuplot - $P/scripts/svm-times.gpi >/dev/null 2>&1 <<EOP
            plot_title  = "$title ($t)"
            plot_output = "$P/svm-plots/$t/$title"
            plot_data   = "$data"
EOP
        if [ $? -ne 0 ]
        then
            printf '!'
        fi
        printf "$t  "
    done
    echo
done

