#PBS -N backtesting 
#PBS -l walltime=06:30:00 
#PBS -l select=1:ncpus=1:mem=10gb


module load anaconda3/personal
source activate RegularisedVAR

export DESIGN_FILE='../../../data/processed/Santander/735logdata775_no555.csv'

cd $PBS_O_WORKDIR
python backtest.py $DESIGN_FILE backtesting_config.yaml
