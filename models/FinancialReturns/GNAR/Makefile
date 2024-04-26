# Makefile for doing GNAR backtesting
# Author: Brendan Martin
# Description: Build processes for:  backtesting GNAR, 
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

# Declare phony targets
.PHONY: clean


# Do Backtesting
predictions-$(ARRAY_INDEX_VALUE).csv backtesting_hyp.txt : $(BACKTEST_DESIGN) GNAR_backtest_config.yaml GNAR-backtest.R adjacency.csv
	Rscript GNAR-backtest.R adjacency.csv GNAR_backtest_config.yaml $(BACKTEST_DESIGN) 
	mkdir -p $(OUTPUT_DIR)  # Create the output directory if it doesn't exist
	scp backtesting_hyp.txt $(OUTPUT_DIR)

# Compute backtesting statistics 
PnL.csv hit.csv long.csv spearman_corr.csv summary_statistics.txt : backtest_statistics.py $(BACKTEST_DESIGN) predictions-$(ARRAY_INDEX_VALUE).csv backtesting_config.yaml
	python backtest_statistics.py $(BACKTEST_DESIGN) predictions-$(ARRAY_INDEX_VALUE).csv backtesting_config.yaml 
	mkdir -p $(OUTPUT_DIR)  # Create the output directory if it doesn't exist
	scp PnL.csv hit.csv long.csv spearman_corr.csv summary_statistics.txt $(OUTPUT_DIR)

# Clean rule to remove generated files and the output directory
clean:
	rm -rf *.txt *.pdf 


