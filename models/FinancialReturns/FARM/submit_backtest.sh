#PBS -N backtesting 
#PBS -l walltime=01:00:00 
#PBS -l select=1:ncpus=16:mem=4gb
#PBS -J 1-16

export NUM_ARRAY_INDICES=16

module load anaconda3/personal
source activate RegularisedVAR

export DESIGN_FILE='../../data/processed/stocks_no_market_cleaned.csv'

cd $PBS_O_WORKDIR
python backtest.py $DESIGN_FILE backtesting_config.yaml

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