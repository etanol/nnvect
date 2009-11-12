#!/bin/sh

P=`dirname $0`/..
for d in sizes times mflops scales
do
    test -d $P/plots/$d || mkdir -p $P/plots/$d
done

if [ -f $P/scripts/datafiles ]
then
    . $P/scripts/datafiles
else
    files=`ls $P/sample_nn_data/*.tst | sed 's/\.tst$//'`
fi

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
    data=$P/plots/sizes/table-$title.txt
    for t in $types
    do
        input=$P/outputs/$title-simple-vec-$t-1
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
        plot_output = "$P/plots/sizes/$title"
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
    data=$P/plots/times/table-$title.txt
    for t in $types
    do
        line="$t"
        for mode in sca vec
        do
            for impl in simple unroll2 unroll4
            do
                input=$P/outputs/$title-$impl-$mode-$t-1
                if [ -f $input ]
                then
                    time=`awk '$2 == "Minimum" { print $4 }' $input`
                else
                    time='-'
                fi
                line="$line  $time"
            done
        done
        echo "$line"
    done >$data
    gnuplot - $P/scripts/times.gpi >/dev/null 2>&1 <<EOP
        plot_title  = "$title"
        plot_output = "$P/plots/times/$title"
        plot_data   = "$data"
EOP
    if [ $? -ne 0 ]
    then
        printf '!'
    fi
    printf 'times  '

    #
    # Plot MFLOPS
    #
    data=$P/plots/mflops/table-$title.txt
    for t in $types
    do
    line="$t"
        for mode in sca vec
        do
            for impl in simple unroll2 unroll4
            do
                input=$P/outputs/$title-$impl-$mode-$t-1
                if [ -f $input ]
                then
                    mflops=`awk '$2 == "Minimum" { print substr($6, 2) }' $input`
                else
                    mflops='-'
                fi
                line="$line  $mflops"
            done
        done
        echo "$line"
    done >$data
    gnuplot - $P/scripts/mflops.gpi >/dev/null 2>&1 <<EOP
        plot_title  = "$title"
        plot_output = "$P/plots/mflops/$title"
        plot_data   = "$data"
EOP
    if [ $? -ne 0 ]
    then
        printf '!'
    fi
    printf 'mflops  '

    #
    # Plot scalability
    #
    data=$P/plots/scales/table-$title.txt
    for threads in 1 2 3 4 5 6 7 8
    do
        line=''
        for t in $types
        do
            for mode in sca vec
            do
                for impl in simple unroll2 unroll4
                do
                    input=$P/outputs/$title-$impl-$mode-$t-$threads
                    if [ -f $input ]
                    then
                        time=`awk '$2 == "Minimum" { print $4 }' $input`
                    else
                        time='-'
                    fi
                    line="$time  $line"
                done
            done
        done
        echo "$threads  $line"
    done | awk '$1 == 1 {
                    printf "1"
                    for (i = 2;  i <= NF;  i++)
                    {
                        printf "  1"
                        base[i] = $i
                    }
                    printf "\n"
                    next
                }
                {
                    printf "%d", $1
                    for (i = 2;  i <= NF;  i++)
                        if (base[i] == "-" || $i == "-")
                            printf "  -"
                        else
                            printf "  %f", base[i] / $i
                    printf "\n"
                }' >$data.txt
    gnuplot - $P/scripts/scales.gpi >/dev/null 2>&1 <<EOP
        plot_title  = "$title"
        plot_output = "$P/plots/scales/$title"
        plot_data   = "$data"
EOP
    if [ $? -ne 0 ]
    then
        printf '!'
    fi
    printf "scales\n"
done

