#!/bin/bash

# This script will create a new directoy with the name `year-month-day hour:mintue`
# and copy over template files for the pickaxe run.
# As Pickaxe runs become more complex, this file may need to be updated

mkdir "$(date +%Y-%m-%d\ %H:%M:%S)"
# cd "$(date +%Y-%m-%d\ %H:%M)"
touch "$(date +%Y-%m-%d\ %H:%M:%S)"/info.txt
cp pickaxe_run_template.py "$(date +%Y-%m-%d\ %H:%M:%S)"
mv "$(date +%Y-%m-%d\ %H:%M:%S)"/pickaxe_run_template.py "$(date +%Y-%m-%d\ %H:%M:%S)"/run_pickaxe.py