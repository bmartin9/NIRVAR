#PBS -N backtesting 
#PBS -l walltime=07:50:00 
#PBS -l select=1:ncpus=1:mem=1gb

module load anaconda3/personal
source activate RegularisedVAR

cd $PBS_O_WORKDIR
python lasso-comparison.py