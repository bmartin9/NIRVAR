# Makefile for building network-informed VAR estimation pipeline
# Author: Brendan Martin
# Description: Build processes for: simulating from the SBM VAR model, backtesting, 
#              comparison of estimated coefficients and ground truth coefficients,
#              computing backtesting statistics 

# Define the user-specified output directory (default to 'output')
OUTPUT_DIR ?= output

# Define the default dependency for the second rule
BACKTEST_DESIGN ?= design_matrix.csv

# Define the default target_feature for the third rule
TARGET_FEATURE ?= 1

# Read in the ARRAY INDEX that we use to name the output predictions-, phi_hat- and labels_hat-
ARRAY_INDEX ?= 1
ARRAY_INDEX_VALUE := $(ARRAY_INDEX)
new_target:
	@echo "ARRAY_INDEX_VALUE is $(ARRAY_INDEX_VALUE)"

# NUM_FILES_CREATED_SO_FAR:=$(find . -maxdepth 1 -type f -name 'predictions-*' | wc -l)
# NUM_FILES_CREATED_SO_FAR:=$(shell find . -maxdepth 1 -type f -name 'predictions-*' 2>/dev/null | wc -l)
# NUM_ARRAY_INDICES:=$(shell echo $$NUM_ARRAY_INDICES)

# Declare phony targets
.PHONY: clean

# Simulate timeseries 
design_matrix.csv phi.csv generating_hyperparameters.txt labels_gt.csv: generate_timeseries.py simulation_config.yaml
	python generate_timeseries.py simulation_config.yaml 
	mkdir -p $(OUTPUT_DIR)  # Create the output directory if it doesn't exist
	scp generating_hyperparameters.txt $(OUTPUT_DIR)

# Do Backtesting
predictions-$(ARRAY_INDEX_VALUE).csv phi_hat-$(ARRAY_INDEX_VALUE).csv backtesting_hyp.txt labels_hat-$(ARRAY_INDEX_VALUE).csv: $(BACKTEST_DESIGN) backtesting_config.yaml backtest.py
	python backtest.py $(BACKTEST_DESIGN) backtesting_config.yaml 
	mkdir -p $(OUTPUT_DIR)  # Create the output directory if it doesn't exist
	scp backtesting_hyp.txt $(OUTPUT_DIR)

# Compare Estimated Coefficients with Ground Truth Coefficients
rmse.pdf ari.pdf : compare_coefficients.py phi.csv phi_hat-$(ARRAY_INDEX_VALUE).csv backtesting_config.yaml labels_gt.csv labels_hat-$(ARRAY_INDEX_VALUE).csv
	python compare_coefficients.py phi.csv phi_hat-$(ARRAY_INDEX_VALUE).csv backtesting_config.yaml labels_gt.csv labels_hat-$(ARRAY_INDEX_VALUE).csv 
	mkdir -p $(OUTPUT_DIR)  # Create the output directory if it doesn't exist
	scp rmse.pdf ari.pdf $(OUTPUT_DIR) 

# Compute backtesting statistics 
PnL.csv hit.csv long.csv spearman_corr.csv summary_statistics.txt : backtest_statistics.py $(BACKTEST_DESIGN) predictions-$(ARRAY_INDEX_VALUE).csv backtesting_config.yaml
	python backtest_statistics.py $(BACKTEST_DESIGN) predictions-$(ARRAY_INDEX_VALUE).csv backtesting_config.yaml 
	mkdir -p $(OUTPUT_DIR)  # Create the output directory if it doesn't exist
	scp PnL.csv hit.csv long.csv spearman_corr.csv summary_statistics.txt $(OUTPUT_DIR)

# Define the default targets that get created when `make` is called with no specified target
all: design_matrix.csv phi.csv generating_hyperparameters.txt predictions-$(ARRAY_INDEX_VALUE).csv phi_hat-$(ARRAY_INDEX_VALUE).csv backtesting_hyp.txt 
	rmse.pdf ari.pdf PnL.csv hit.csv long.csv spearman_corr.csv summary_statistics.txt

# Clean rule to remove generated files and the output directory
clean:
	rm -rf *.csv *.txt *.pdf 


