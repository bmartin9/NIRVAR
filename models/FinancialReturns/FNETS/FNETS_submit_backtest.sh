#PBS -N backtesting 
#PBS -l walltime=04:30:00 
#PBS -l select=1:ncpus=95:mem=9gb
#PBS -J 1-95

export NUM_ARRAY_INDICES=95

module load anaconda3/personal
source activate RegularisedVAR

export DESIGN_FILE='../../data/processed/stocks_no_market_cleaned.csv'
export OUTPUT_DIRECTORY='0.5-fnets-unrestricted-ic-504'
cd $PBS_O_WORKDIR
make predictions-$PBS_ARRAY_INDEX.csv BACKTEST_DESIGN=$DESIGN_FILE ARRAY_INDEX=$PBS_ARRAY_INDEX OUTPUT_DIR=$OUTPUT_DIRECTORY

NUM_FILES_CREATED_SO_FAR=$(find . -maxdepth 1 -type f -name 'predictions-*' 2>/dev/null | wc -l)
echo $NUM_FILES_CREATED_SO_FAR 
if [ $NUM_FILES_CREATED_SO_FAR -eq $NUM_ARRAY_INDICES ]; then
        if ls predictions-*.csv 1>/dev/null 2>&1; then
                cat $(ls -v predictions-*.csv) > predictions.csv
                cat $(ls -v factors-*.csv) > factors.csv
                rm predictions-*.csv
                rm factors-*.csv
        else
                echo "File predictions-*.csv does not exist."
        fi

fi


scp predictions.csv backtesting_hyp.txt  $OUTPUT_DIRECTORY