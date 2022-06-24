#!/bin/bash

for d in $(find /Users/mgiammar/Documents/MINE-Database/pickaxe_runs/cluster_1gen -maxdepth 1 -type d)
do
  #Do something, the directory is accessible with $d:
  if [ -f "$d/run_pickaxe.py" ]; then
    echo $d
    python "$d/run_pickaxe.py"
  fi
done