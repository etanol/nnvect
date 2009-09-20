#!/bin/sh

RFILE=res
FPREFIX=fail

#
# Verification function.  Arguments are:
#
#     $1 --> executed binary
#     $2 --> test data
#     $3 --> data type
#     $4 --> scalar (optional)
#
check()
{
    if [ $? -ne 0 ]
    then
        echo "NOT COMPLETED"
    elif cmp -s result-$2 $RFILE
    then
        echo "OK"
    else
        dumpfile=$FPREFIX-$2-$1-$3${4+-scalar}
        ../debug/$1 --runs=1 --type=$3 ${4+--scalar} $2 2>&1 | awk -f ../debug/filter.awk >$dumpfile
        echo "FAILED --> $dumpfile"
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

    echo "Testing '$1' with '$2'"
    echo
    printf "  Euclidean distance "
    for t in "byte  "  "short "  "int   "  "float "  "double"
    do
        printf "$t "
        printf "scalar  "
        ../$1 --output=$RFILE --type=$t --scalar $2 >/dev/null 2>&1
        check $1 $2 $t scalar
        printf "                            vector  "
        ../$1 --output=$RFILE --type=$t $2 >/dev/null 2>&1
        check $1 $2 $t
        printf "                     "
    done
    echo
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
run_test unroll2 basic
run_test unroll2 large
run_test unroll4 basic
run_test unroll4 large

rm -f $RFILE
