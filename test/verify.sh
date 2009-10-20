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
#     $4 --> scalar or vector
#     $5 --> block size (optional)
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
        dumpfile=$FPREFIX-$2-$1-$3.$4 mv $RFILE $dumpfile
        FAILED="$FAILED $dumpfile"
        printf '!'
    fi
}


#
# Test function.  Arguments are:
#
#     $1 --> binary to execute
#     $2 --> test data to load and use
#     $3 --> size of the block (optional)
#
run_test()
{
    for t in "byte  "  "short "  "int   "  "float "  "double"
    do
        ../$1 --output=$RFILE --type=$t --scalar $2 >/dev/null 2>&1
        check $1 $2 $t scalar
        ../$1 --output=$RFILE --type=$t $2 >/dev/null 2>&1
        check $1 $2 $t vector
    done
    if [ -n "$3" ]
    then
        for t in "byte  "  "short "  "int   "  "float "  "double"
        do
            ../$1 --output=$RFILE --blocksize=$3 --type=$t --scalar $2 >/dev/null 2>&1
            check $1 $2 $t scalar $3
            ../$1 --output=$RFILE --blocksize=$3 --type=$t $2 >/dev/null 2>&1
            check $1 $2 $t vector $3
        done
    fi
}


rm -f $FPREFIX-*

echo "====  BUILDING  ===="
make -C ..

echo
echo
echo "====  TESTING  ===="
run_test simple basic
run_test simple large  1000
run_test unroll2 basic
run_test unroll2 large 1000
run_test unroll4 basic
run_test unroll4 large 1000
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
