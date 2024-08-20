#PBS -N backtesting 
#PBS -l walltime=07:50:00 
#PBS -l select=1:ncpus=1:mem=15gb

module load anaconda3/personal
source activate RegularisedVAR

cd $PBS_O_WORKDIR
python FRED_sparsity_variance.py ../../../data/processed/FRED-MD/fred-balanced.csv ../../FRED-MD/NIRVAR/labels_corr.csv 

