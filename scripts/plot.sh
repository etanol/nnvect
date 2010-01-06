#!/bin/sh

. ./paths.sh

#
# AWK scripts
#
HAS_FLOATS='$1 == "floats" { print $2 }'
NN_SIZE='/^ +Data array is/ { printf "  %d", $4 }'
NN_TIME='/^- Minimum/ { printf $4 }'
NN_MOPS='/^- Minimum/ { printf substr($6, 2) }'
SVM_TIMES='/^TRAINING TIME:/ { train = $3 }
           /^PREDICT TIME:/  { predict = $3 }
           END { printf "%f  %f", predict, train }'

#
# Parameters: {plot_script} {title} {input} {output}
do_plot()
{
    gnuplot - $1 >/dev/null 2>&1 <<EOP
        plot_title  = "$2"
        plot_data   = "$3"
        plot_output = "$4"
EOP
}

ensure_path $plots
for d in sizes times mflops scales comptimes comptimes2 gputimes
do
    ensure_path $plots/$d
done


for file in $nn_inputs/*.tst
do
    f=${file%.tst}
    bf=${f#$nn_inputs/}
    has_floats=$(awk "$HAS_FLOATS" $f.tst.info)
    if [ $has_floats = yes ]
    then
        types='float double'
        tk='float'
    else
        types='byte short int float double'
        tk='byte'
    fi

    #
    # Plot sizes
    #
    printf "  $bf sizes ... "
    data=$plots/sizes/table-$bf.txt
    for t in $types
    do
        input=$nn_outputs/$bf-unroll4-vec-$t-b2-1
        if [ -f $input ]
        then
            values=$(awk "$NN_SIZE" $input)
        else
            values='  -  -'
        fi
        echo "$t$values"
    done >$data
    do_plot sizes.gpi $bf $data $plots/sizes/$bf
    echo done

    #
    # Plot times
    #
    printf "  $bf times ... "
    data=$plots/times/table-$bf.txt
    for t in $types
    do
        line="$t"
        for mode in sca vec
        do
            for impl in simple unroll2 unroll4
            do
                for block in 0 1 2
                do
                    input=$nn_outputs/$bf-$impl-$mode-$t-b$block-1
                    if [ -f $input ]
                    then
                        time=$(awk "$NN_TIME" $input)
                    else
                        time='-'
                    fi
                    line="$line  $time"
                done
            done
        done
        echo "$line"
    done >$data
    do_plot times.gpi $bf $data $plots/times/$bf
    echo done

    #
    # Plot MFLOPS
    #
    printf "  $bf mflops ... "
    data=$plots/mflops/table-$bf.txt
    for t in $types
    do
        line="$t"
        for mode in sca vec
        do
            for impl in simple unroll2 unroll4
            do
                for block in 0 1 2
                do
                    input=$nn_outputs/$bf-$impl-$mode-$t-b$block-1
                    if [ -f $input ]
                    then
                        mflops=$(awk "$NN_MOPS" $input)
                    else
                        mflops='-'
                    fi
                    line="$line  $mflops"
                done
            done
        done
        echo "$line"
    done >$data
    do_plot mflops.gpi $bf $data $plots/mflops/$bf
    echo done

    #
    # Plot scalability
    #
    printf "  $bf scalability ... "
    data=$plots/scales/table-$bf.txt
    for threads in 1 2 3 4 5 6 7 8
    do
        line=''
        for t in $types
        do
            for mode in sca vec
            do
                for impl in simple unroll2 unroll4
                do
                    input=$nn_outputs/$bf-$impl-$mode-$t-b1-$threads
                    if [ -f $input ]
                    then
                        time=$(awk "$NN_TIME" $input)
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
                }' >$data
    do_plot scales.gpi $bf $data $plots/scales/$bf
    echo done

    #
    # Plot time comparisons
    #
    printf "  $bf time comparisons ... "
    for t in $types
    do
        in_svm=$svm_outputs/${bf%.scale}.scale
        in_lin=$lin_outputs/$bf
        in_worst=$nn_outputs/$bf-simple-sca-$t-b0-1
        in_best=$nn_outputs/$bf-unroll4-vec-$t-b2-1
        svm='-  -'
        lin='-  -'
        worst='-'
        best='-'
        test -f $in_svm && svm=$(awk "$SVM_TIMES" $in_svm)
        test -f $in_lin && lin=$(awk "$SVM_TIMES" $in_lin)
        test -f $in_worst && worst=$(awk "$NN_TIME" $in_worst)
        test -f $in_best && best=$(awk "$NN_TIME" $in_best)
        data=$plots/comptimes/table-$bf-$t.txt
        {
            echo "LibSVM  $svm"
            echo "LibLINEAR  $lin"
            echo "NN-worst  $worst  -"
            echo "NN-best  $best  -"
        } >$data
        do_plot comptimes.gpi "$bf ($t)" $data $plots/comptimes/$bf-$t
        data=$plots/comptimes2/table-$bf-$t.txt
        {
            echo "LibLINEAR  $lin"
            echo "NN-best  $best  -"
        } >$data
        do_plot comptimes.gpi "$bf ($t)" $data $plots/comptimes2/$bf-$t
    done
    echo done

    #
    # Plot NN vs GPU time comparisons
    #
    printf "  $bf GPU vs NN ... "
    in_worst=$nn_outputs/$bf-simple-sca-float-b0-1
    in_best=$nn_outputs/$bf-unroll4-vec-float-b0-1
    in_best8=$nn_outputs/$bf-unroll4-vec-float-b0-8
    in_gpu=$cl_outputs/$bf
    worst='-'
    best='-'
    best8='-'
    gpu='-'
    test -f $in_worst && worst=$(awk "$NN_TIME" $in_worst)
    test -f $in_best && best=$(awk "$NN_TIME" $in_best)
    test -f $in_best8 && best8=$(awk "$NN_TIME" $in_best8)
    test -f $in_gpu && gpu=$(awk "$NN_TIME" $in_gpu)
    data=$plots/gputimes/table-$bf.txt
    if :
    then
        echo "NN-worst  $worst"
        echo "NN-best  $best"
        echo "GPU  $gpu"
        echo "NN-best8 $best8"
    fi >$data
    do_plot comptimes.gpi "$bf" $data $plots/gputimes/$bf
    echo done

done

