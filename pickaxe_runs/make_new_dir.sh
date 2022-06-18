#!/bin/bash

# This script will create a new directoy with the name provided as the first argument,
# otherwise a default of `year-month-day hour:mintue`,
# and copy over template files for the pickaxe run.
# As Pickaxe runs become more complex, this file may need to be updated

dir_name=${1:-"$(date +%Y-%m-%d_%H:%M:%S)"}  # Use first arg as name, otherwise current datetime
echo "Creating directory ${dir_name}"

mkdir $dir_name
# touch $dir_name/info.txt
cp pickaxe_run_template.py "${dir_name}/"
mv "${dir_name}/pickaxe_run_template.py" "${dir_name}/run_pickaxe.py"

# mkdir "$(date +%Y-%m-%d\ %H:%M:%S)"
# # cd "$(date +%Y-%m-%d\ %H:%M)"
# touch "$(date +%Y-%m-%d\ %H:%M:%S)"/info.txt
# cp pickaxe_run_template.py "$(date +%Y-%m-%d\ %H:%M:%S)"
# mv "$(date +%Y-%m-%d\ %H:%M:%S)"/pickaxe_run_template.py "$(date +%Y-%m-%d\ %H:%M:%S)"/run_pickaxe.py