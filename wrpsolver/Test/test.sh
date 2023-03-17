#!/bin/zsh
i=1
result=`echo $?`
while ((result == 0))
do
    python3 polygons_coverage.py -e 20 -i 10 -c 0.95
    result=`echo $?`
    echo $result >> ./test_result1.txt
    ((i++))
done