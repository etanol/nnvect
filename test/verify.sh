#!/bin/sh

RFILE=res
FPREFIX=fail

#
# Verification function.  Arguments are:
#
#     $1 --> executed binary
#     $2 --> test data
#     $3 --> distance type
#     $4 --> data type
#     $5 --> scalar (optional)
#
check()
{
    if [ $? -ne 0 ]
    then
        echo "NOT COMPLETED"
    elif cmp -s result-$2.$3 $RFILE
    then
        echo "OK"
    else
        dumpfile=$FPREFIX-$2-$1-$4${5+-scalar}.$3
        if [ $3 = "manhattan" ]
        then
                m=--manhattan
        else
                m=
        fi
        ../debug/$1 --runs=1 $m --type=$4 ${5+--scalar} $2 2>&1 | awk -f ../debug/filter.awk >$dumpfile
        echo "FAILED (dump at $dumpfile)"
    fi
}


#
# Test function.  Arguments are:
#
#     $1 --> binary to execute
#     $2 --> test data to load and use
#
run_test()
{

    echo "Testing '$1' with '$2' data"
    echo
    printf "  Euclidean distance test "
    for t in "byte  "  "short "  "int   "  "float "  "double"
    do
        printf "$t "
        printf "scalar  "
        ../$1 --output=$RFILE --type=$t --scalar $2 >/dev/null
        check $1 $2 euclid $t scalar
        printf "                                 vector  "
        ../$1 --output=$RFILE --type=$t $2 >/dev/null
        check $1 $2 euclid $t
        printf "                          "
    done
    echo
    printf "  Manhattan distance test "
    for t in "byte  "  "short "  "int   "  "float "  "double"
    do
        printf "$t "
        printf "scalar  "
        ../$1 --manhattan --output=$RFILE --type=$t --scalar $2 >/dev/null
        check $1 $2 manhattan $t scalar
        printf "                                 vector  "
        ../$1 --manhattan --output=$RFILE --type=$t $2 >/dev/null
        check $1 $2 manhattan $t
        printf "                          "
    done
    echo
}



rm -f $FPREFIX-*

echo "====  BUILDING  ===="
make -C ..
make -C ../debug

echo
echo
echo
echo "====  TESTING  ===="
run_test simple basic
run_test simple large
run_test unroll4 basic
run_test unroll4 large

rm -f $RFILE
