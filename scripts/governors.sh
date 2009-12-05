#!/bin/sh

sys=/sys/devices/system/cpu

if [ $# -gt 0 ]
then
    if fgrep -q $1 $sys/cpu0/cpufreq/scaling_available_governors
    then
        echo Using governor $1
    else
        echo Governor $1 does not exist
        echo
        echo Available governors are:
        echo
        for gov in $(cat $sys/cpu0/cpufreq/scaling_available_governors)
        do
            echo "    $gov"
        done
        exit 2
    fi

    for cpu in $sys/cpu?
    do
        echo $1 > $cpu/cpufreq/scaling_governor
    done
else
    for cpu in $sys/cpu?
    do
        gov=$(cat $cpu/cpufreq/scaling_governor)
        echo $(basename $cpu): $gov
    done
fi

