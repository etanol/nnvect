#!/bin/sh

RFILE=res
FPREFIX=fail
FAILED=''

#
# Verification function.  Arguments are:
#
#     $1 --> executed binary
#     $2 --> test data
#     $3 --> data type
#     $4 --> identificative tag
#
check()
{
    if [ $? -ne 0 ]
    then
        echo "NOT COMPLETED"
    elif cmp -s result-$2 $RFILE
    then
        printf '.'
    else
        dumpfile=$FPREFIX-$2-$1-$3.$4
        mv $RFILE $dumpfile
        FAILED="$FAILED $dumpfile"
        printf '!'
    fi
}


#
# Test function for 1-NN.  Arguments are:
#
#     $1 --> binary to execute
#     $2 --> test data to load and use
#
run_test()
{
    for t in byte short int float double
    do
        ../$1 --output=$RFILE --type=$t --scalar $2 >/dev/null 2>&1
        check $1 $2 $t scalar
        ../$1 --output=$RFILE --type=$t $2 >/dev/null 2>&1
        check $1 $2 $t vector
    done
}


#
# Test function for k-NN.  Arguments are:
#
#     $1 --> value for k (number of neighbours to use)
#     $2 --> test data to load and use
#
knn_test()
{
    for t in byte short int float double
    do
        ../simple --output=$RFILE --type=$t --neighbours=$1 $2 >/dev/null 2>&1
        check simple $2 $t knn
    done
}


rm -f $FPREFIX-*

echo "====  BUILDING  ===="
make -C ..

echo
echo
echo "====  TESTING  ===="
run_test simple basic
run_test simple large
run_test unroll2 basic
run_test unroll2 large
run_test unroll4 basic
run_test unroll4 large
knn_test 2 basic
knn_test 3 large
knn_test 5 large
knn_test 7 large
echo

if [ -n "$FAILED" ]
then
    echo 'The following test failed:'
    for t in $FAILED
    do
        echo "    $t"
    done
else
    echo 'All tests passed'
fi

rm -f $RFILE
