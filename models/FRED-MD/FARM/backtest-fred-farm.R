# Backtest FARM model on FRED data set.

#!/usr/bin/env Rscript
# USAGE: Rscript backtest-fred-farm.R  backtesting_config.yaml <DESIGN_MATRIX>.csv 

library("FarmSelect") 

# Check if the correct number of arguments is provided
if (length(commandArgs(trailingOnly = TRUE)) < 2) {
    stop("USAGE: Rscript backtest-fred-farm.R  backtesting_config.yaml <DESIGN_MATRIX>.csv")
}

# Get the config file path from the second argument
config_file <- commandArgs(trailingOnly = TRUE)[1]

# Read the YAML file
yaml_data <- yaml::read_yaml(config_file)

# Access individual variables and assign them to R variables
SEED <- yaml_data$SEED
n_backtest_days_total <- yaml_data$n_backtest_days
first_prediction_day <- yaml_data$first_prediction_day

###### ENVIRONMENT VARIABLES ###### 
# PBS_ARRAY_INDEX <- as.numeric(Sys.getenv("PBS_ARRAY_INDEX"))
# NUM_ARRAY_INDICES <- as.numeric(Sys.getenv("NUM_ARRAY_INDICES")) 
PBS_ARRAY_INDEX = 1
NUM_ARRAY_INDICES = 239

# Re-define n_backtest_days to be total number of backtesting days divided by the number of array indices 
n_backtest_days <- as.integer(n_backtest_days_total/NUM_ARRAY_INDICES) 
print(n_backtest_days_total)

# Get a list of days to do backtesting on
days_to_backtest <- 1:n_backtest_days + (n_backtest_days * (PBS_ARRAY_INDEX - 1)) - 1 

# Get the design matrix file path from the second argument
design_matrix_file <- commandArgs(trailingOnly = TRUE)[2]

# Read the design matrix
Xs <- read.csv2(design_matrix_file,sep=",",dec = ".",header=TRUE)

# Get dimensions
T <- nrow(Xs)
N <- ncol(Xs) 

s <- matrix(0, nrow = n_backtest_days, ncol = 1)

factors <- matrix(0, nrow = n_backtest_days, ncol = 1)

for (index in 1:n_backtest_days){
    t <- days_to_backtest[[index]] 
    todays_date <- first_prediction_day + t
    print(todays_date)
    furthest_lookback_day <- todays_date - 479
    X_rolled <- Xs[furthest_lookback_day:(todays_date-1),] 
    X_rolled <- data.matrix(X_rolled)
    print(typeof(X_rolled))
    targets <- Xs[(furthest_lookback_day+1):(todays_date),5] # predict INDPRO 
    targets <- data.matrix(targets) 
    print(dim(X_rolled))
    print(dim(targets)) 
    start_time <- Sys.time()
    fit_farm = farm.select(X_rolled,targets,robust = FALSE) #robust, no cross-validation
    full_beta = rep(0,N) 
    full_beta[fit_farm$beta.chosen] = fit_farm$coef.chosen
    print(full_beta[1:10])
    y_pred = full_beta%*%(as.numeric(Xs[todays_date,]))
    s[index] <- y_pred 
    print(fit_farm$nfactors)
    factors[index] <- fit_farm$nfactors
    end_time <- Sys.time()
    time_taken <- end_time - start_time
    cat(sprintf("Time taken: %.2f seconds", time_taken))
}

###### SAVE TO FILE ######
predictions_path <- sprintf("predictions-%d.csv", PBS_ARRAY_INDEX)
# s_matrix <- do.call(rbind, s)
write.table(s, file = predictions_path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

factors_path <- sprintf("factors-%d.csv", PBS_ARRAY_INDEX)
# f_matrix <- do.call(rbind, factors) 
write.table(factors, file = factors_path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

# Write configuration variables to a text file
output_file <- "backtesting_hyp.txt"
f <- file(output_file, "w")

writeLines("{", f)
writeLines(paste0("'SEED':'", SEED, "'"), f)
writeLines(paste0("'n_backtest_days_total':'", n_backtest_days_total, "'"), f)
writeLines(paste0("'first_prediction_day':'", first_prediction_day, "'"), f)
writeLines("}", f)

close(f)