#PBS -N backtesting 
#PBS -l walltime=01:30:00 
#PBS -l select=1:ncpus=75:mem=8gb
#PBS -J 1-75

export NUM_ARRAY_INDICES=75

module load anaconda3/personal
source activate RegularisedVAR

export DESIGN_FILE='../../data/processed/stocks_no_market_cleaned.csv'
export OUTPUT_DIRECTORY='0.5-corr-2features-2d'
cd $PBS_O_WORKDIR
make predictions-$PBS_ARRAY_INDEX.csv BACKTEST_DESIGN=$DESIGN_FILE ARRAY_INDEX=$PBS_ARRAY_INDEX OUTPUT_DIR=$OUTPUT_DIRECTORY

NUM_FILES_CREATED_SO_FAR=$(find . -maxdepth 1 -type f -name 'predictions-*' 2>/dev/null | wc -l)
echo $NUM_FILES_CREATED_SO_FAR 
if [ $NUM_FILES_CREATED_SO_FAR -eq $NUM_ARRAY_INDICES ]; then
        if ls predictions-*.csv 1>/dev/null 2>&1; then
                cat $(ls -v predictions-*.csv) > predictions.csv
                cat $(ls -v phi_hat-*.csv) > phi_hat.csv
                cat $(ls -v labels_hat-*.csv) > labels_hat.csv
                rm predictions-*.csv
                rm phi_hat-*.csv
                rm labels_hat-*.csv
        else
                echo "File predictions-*.csv does not exist."
        fi

fi


scp predictions.csv phi_hat.csv labels_hat.csv  backtesting_hyp.txt  $OUTPUT_DIRECTORY