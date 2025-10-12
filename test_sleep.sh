#!/bin/bash
echo "[test_sleep] start PID=$$"
for i in $(seq 1 10000); do
    echo "[test_sleep] i=$i"
    sleep 2
done

