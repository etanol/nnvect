/\[DEBUG.+ \.$/ {
        a = NF - 3
        b = NF - 2
        c = NF - 1
        printf "%s\t%s\t%s\n", $a, $b, $c
}
