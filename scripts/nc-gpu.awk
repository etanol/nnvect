#!/usr/bin/awk -f

BEGIN {
    hz = 1242000000
    x = 0
}

$4 == "elements" {
    if (x == 0)
        trn = $3
    else if (x == 1)
        tst = $3
    x++
}

$4 == "dimensions," { dim = $3 }

$2 == "Minimum" { time = $4 }

END {
    printf "%.4f\n", time * hz / (dim * trn * tst)
}

