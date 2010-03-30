#!/bin/sh

exit_code=0
case $(uname) in
    Linux)
        # Look for cpufreq first (is more accurate)
        sys='/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq'
        proc='/proc/cpuinfo'
        if [ -f $sys ]
        then
            khz=$(cat $sys)
            hz=$(($khz * 1000))
            comment='Frequency accurately obtained from cpufreq scaling information'
        elif [ -f $proc ]
        then
            hz=$(awk '$2 == "MHz" { printf "%d", $4 * 1000000; exit }' $proc)
            comment='Frequency unaccurately obtained from /proc/cpuinfo'
            echo "WARNING: The frequency estimation may not be accurate, please review file $1"
        else
            echo "Sorry, I don't know how to get the CPU frequency"
            echo "Please edit the file $1 by hand"
        fi
        ;;
    *)
        echo "Sorry, I don't know how to get the CPU frequency"
        echo "Please edit the file $1 by hand"
        ;;
esac

if [ -z "$hz" ]
then
    hz='{FILL_IN}'
    header='Replace {FILL_IN} with the frequency in Hz'
    exit_code=1
fi

cat >$1 <<FREQUENCY
/*
 * CPU Frequency information.  Separated from the rest of the code for automatic
 * generation, when possible.
 */

/* $comment */
#ifndef CPU_HZ
#define CPU_HZ  $hz
#endif
FREQUENCY

exit $exit_code

