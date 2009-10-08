#!/bin/sh

P=`dirname $0`/..
test -d $P/plots || mkdir $P/plots

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

    #
    # Plot sizes
    #
    data=$P/plots/sizes-$title.txt
    for t in $types
    do
        input=$P/outputs/$title-simple-vec-b0-$t
        if [ -f $input ]
        then
            values=`awk '/^ +Data array is/ { printf "  " $4 }' $input`
        else
            values='- -'
        fi
        echo "$t$values"
    done >$data
    gnuplot - $P/scripts/sizes.gpi >/dev/null 2>&1 <<EOP
        plot_title  = "$title"
        plot_output = "$P/plots/$title"
        plot_data   = "$data"
EOP
    if [ $? -ne 0 ]
    then
        printf '!'
    fi
    printf 'sizes  '

    #
    # Plot times
    #
    data=$P/plots/times-$title.txt
    for t in $types
    do
        line="$t"
        for bmegs in 0 4
        do
            for mode in sca vec
            do
                for impl in simple unroll2 unroll4
                do
                    input=$P/outputs/$title-$impl-$mode-b$bmegs-$t
                    if [ -f $input ]
                    then
                        time=`awk '$2 == "Minimum" { print $4 }' $input`
                    else
                        time='-'
                    fi
                    line="$line  $time"
                done
            done
        done
        echo "$line"
    done >$data
    gnuplot - $P/scripts/times.gpi >/dev/null 2>&1 <<EOP
        plot_title  = "$title"
        plot_output = "$P/plots/$title"
        plot_data   = "$data"
EOP
    if [ $? -ne 0 ]
    then
        printf '!'
    fi
    printf "times\n"
done

