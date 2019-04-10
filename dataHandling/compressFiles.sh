#!/bin/bash

################################################
# Each month takes about of 15 min to compress on laptop which implies that total runtime
# approx 15*12*6 = 1080 = 18 hours

# Folder where to save the compressed files
endFolder=""

suffix=".nc"

# Data source
pathEC=""
pathMS=""
pathRA=""

varEC=("cape" "tp" "u700" "u850" "v700" "v850")
varMS=("hcc" "lcc" "mcc" "prec1h" "t" "tcc" "tiw")
varRA=("radar")

yearsData=("2010" "2011" "2012" "2013" "2014" "2015")
nMonths=12

filesYear=()

for j in "${yearsData[@]}"; do

	for (( i = 1; i <= $nMonths; i++ )); do

		if (($i < 10)); then
			datestring="$j"0"$i"
		else
			datestring="$j$i"
		fi

		# EC vars
		for ec in "${varEC[@]}"; do
			atmVarPath=$pathEC"ec_"$ec"_"$datestring$suffix
			filesYear+=($atmVarPath)
		done

		# mesan vars
		for ms in "${varMS[@]}"; do
			atmVarPath=$pathMS"mesan_"$ms"_"$datestring$suffix
			filesYear+=($atmVarPath)
		done

		# radar
		atmVarPath=$pathRA"radar_"$datestring$suffix
		filesYear+=($atmVarPath)

		commandString=""

		for f in "${filesYear[@]}"; do
			commandString=$commandString" "$f
		done

		tar --absolute-names -czvf $endFolder"dataTAR_"$datestring.tar.gz $commandString

		filesYear=()

	done

done