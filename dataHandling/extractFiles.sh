#!/bin/bash

# Construct script that extract the files for the validation and training set
yearsData=("2010" "2011" "2012" "2013" "2014" "2015")

startMonth=1
endMonth=12

# Data source
pathData=""

# Save folder
pathDataEnd=""

for j in "${yearsData[@]}"; do

	for (( i = $startMonth; i <= $endMonth; i++ )); do

		if (($i < 10)); then
			datestring="$j"0"$i"
		else
			datestring="$j$i"
		fi

		atmCommand=$pathData"dataTAR_"$datestring".tar.gz"

		tar -xvzf $atmCommand -C $pathDataEnd

	done

done