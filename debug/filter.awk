$1 == "[DEBUG" && $7 == "." {
        printf "%s\t%s\t%s\n", $4, $5, $6
}
