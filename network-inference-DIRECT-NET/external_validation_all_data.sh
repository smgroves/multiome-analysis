#!/bin/bash

# to use this code, make sure the line below in main_all_data.py that takes in command line arguments is not commented out
# sample = sys.argv[1]

for sample in "RU1065" "RU1066" "RU1080" "RU1108" "RU1124" "RU1144" "RU1145" "RU1152" "RU1181" "RU1195" "RU1215" "RU1229" "RU1231" "RU1293" "RU1311" "RU1322"
do
  echo "$sample"
  python main_all_data.py "$sample"
done

