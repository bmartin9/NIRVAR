#PBS -N backtesting 
#PBS -l walltime=01:30:00 
#PBS -l select=1:ncpus=30:mem=20gb
#PBS -J 1-30

export NUM_ARRAY_INDICES=30

module load anaconda3/personal
source activate RegularisedVAR

export DESIGN_FILE='../../../data/processed/Santander/735logdata775_no555.csv'

cd $PBS_O_WORKDIR
Rscript backtest.R 735GNAR_adjacency_R3.csv backtesting_config.yaml $DESIGN_FILE

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

# python backtest_statistics.py $DESIGN_FILE predictions-$PBS_ARRAY_INDEX.csv backtesting_config.yaml 

# python 0.3-visualise-backtesting.py PnL.csv

# scp predictions.csv factors.csv  backtesting_hyp.txt  $OUTPUT_DIRECTORY
