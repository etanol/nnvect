#!/bin/sh

P=`dirname $0`/..
test -d $P/plots || mkdir $P/plots

table_header='Type:Scalar simple:Scalar unroll2:Scalar unroll4:Vectorized simple:Vectorized unroll2:Vectorized unroll4:x:x:x:x:x:x:'
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

    name=`basename $f`
    table=$P/plots/table-$name.txt
    printf "$table_header\n" >$table

    for t in $types
    do
        printf "$t"
        for bmegs in 0 4
        do
            for mode in sca vec
            do
                for impl in simple unroll2 unroll4
                do
                    output=$P/outputs/$name-$impl-$mode-b$bmegs-$t
                    if [ -f $output ]
                    then
                        time=`awk '$2 == "Minimum" { print $4 }' $output`
                    else
                        time='-'
                    fi
                    printf ":$time"
                done
            done
        done
        printf "\n"
    done >>$table

    gnuplot - $P/scripts/histograms.gpi <<EOP
        plot_title  = "$name"
        plot_output = "$P/plots/$name"
        plot_data   = "$table"
EOP
done
