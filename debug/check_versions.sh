#!/bin/sh

#
# Euclid versions
#
for type in byte short int float double
do
        echo -n "Verifying Euclid, scalar, $type... "
        ./bench --runs 1 --type $type --scalar ../test/basic 2>&1 | \
        awk -f filter.awk >result_check

        if cmp -s ../test/debug.euclid result_check
        then
                echo "OK"
        else
                echo "FAILED"
        fi

        echo -n "Verifying Euclid, vectorized, $type... "
        ./bench --runs 1 --type $type ../test/basic 2>&1 | \
        awk -f filter.awk >result_check

        if cmp -s ../test/debug.euclid result_check
        then
                echo "OK"
        else
                echo "FAILED"
        fi
done

#
# Manhattan versions
#
for type in byte short int
do
        echo -n "Verifying Manhattan, scalar, $type... "
        ./bench --manhattan --runs 1 --type $type --scalar ../test/basic 2>&1 | \
        awk -f filter.awk >result_check

        if cmp -s ../test/debug.manhattan result_check
        then
                echo "OK"
        else
                echo "FAILED"
        fi

        echo -n "Verifying Manhattan, vectorized, $type... "
        ./bench --manhattan --runs 1 --type $type ../test/basic 2>&1 | \
        awk -f filter.awk >result_check

        if cmp -s ../test/debug.manhattan result_check
        then
                echo "OK"
        else
                echo "FAILED"
        fi
done

