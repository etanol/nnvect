#!/bin/sh

#
# Check function.  Arguments are:
#
#       $1 -> euclid | manhattan
#       $2 -> integers | floats
#       $3 -> scalar | vector
#
check()
{
        case $1 in
                euclid)
                        label1="   euclid"
                        manh=""
                        ;;
                manhattan)
                        label1="manhattan"
                        manh="--manhattan"
                        ;;
        esac
        case $2 in
                integers) types="byte short int" ;;
                floats) types="float double" ;;
        esac
        case $3 in
                scalar) sca="--scalar" ;;
                vector) sca="" ;;
        esac

        for type in $types
        do
                echo -n "Verifying $label1, $3, $type	... "
                ./bench --runs 1 $sca $manh --type $type ../test/basic 2>&1 | awk -f filter.awk >result_check

                if cmp -s ../test/debug-$2.$1 result_check
                then
                        echo "OK"
                else
                        echo "FAILED"
                fi
        done
}


check euclid integers scalar
check euclid floats scalar
check manhattan integers scalar
check manhattan floats scalar

check euclid integers vector
check euclid floats vector
check manhattan integers vector
check manhattan floats vector

