# Backtest FNETS model on hospital data.

#!/usr/bin/env Rscript
# USAGE: Rscript backtest.R  backtesting_config.yaml <DESIGN_MATRIX>.csv 

library("fnets") 

# Check if the correct number of arguments is provided
if (length(commandArgs(trailingOnly = TRUE)) != 2) {
    stop("USAGE: Rscript FNETS-backtest.R backtesting_config.yaml <DESIGN_MATRIX>.csv")
}

# Get the config file path from the second argument
config_file <- commandArgs(trailingOnly = TRUE)[1]

# Read the YAML file
yaml_data <- yaml::read_yaml(config_file)

# Access individual variables and assign them to R variables
SEED <- yaml_data$SEED
n_backtest_days_total <- yaml_data$n_backtest_days
first_prediction_day <- yaml_data$first_prediction_day
Q <- yaml_data$Q 
target_feature <- yaml_data$target_feature 
lookback_window <- yaml_data$lookback_window

###### ENVIRONMENT VARIABLES ###### 
# PBS_ARRAY_INDEX <- as.numeric(Sys.getenv("PBS_ARRAY_INDEX"))
# NUM_ARRAY_INDICES <- as.numeric(Sys.getenv("NUM_ARRAY_INDICES")) 
PBS_ARRAY_INDEX = 1
NUM_ARRAY_INDICES = 1

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
N_times_Q <- ncol(Xs)

N <- N_times_Q / Q

s <- list()

factors <- list()

for (index in 1:n_backtest_days){
    t <- days_to_backtest[[index]] 
    todays_date <- first_prediction_day + t
    print(todays_date)
    furthest_lookback_day <- todays_date - lookback_window
    X_train <- Xs[1:(todays_date),] 
    X_train <- data.matrix(X_train)
    print(dim(X_train))
    # X_train_diff <- diff(X_train) 
    fit_fnets <- fnets(x=X_train,
                    center = FALSE,
                    fm.restricted = TRUE,
                    q = 1,
                    var.order = 1,
                    var.method = "lasso",
                    do.threshold = FALSE,
                    do.lrpc = FALSE,
                    ) 
    if (fit_fnets$q == 0){ 
      fit_fnets <- fnets(x=X_train,
                    center = FALSE,
                    fm.restricted = TRUE,
                    q = 1,
                    var.order = 1,
                    var.method = "lasso",
                    do.threshold = FALSE,
                    do.lrpc = FALSE,
                    ) 
    }
    print(fit_fnets$q)
    print(object.size(fit_fnets)) 
    factors[[index]] <- fit_fnets$q 
    pr <- predict(fit_fnets, n.ahead = 1,fc.restricted = TRUE)
    predictions_matrix <- pr$forecast  
    # Check if all elements in the matrix are 0
    if(all(predictions_matrix == 0)) {
      print("The predictions matrix contains all 0s.") 
      print(fit_fnets) 
    }  
    # s[[index]] <- predictions_matrix[1:N] + X_train[(lookback_window + index - 1),]
    s[[index]] <- predictions_matrix[1:N] 
    rm(fit_fnets)
}

###### SAVE TO FILE ######
predictions_path <- sprintf("predictions-%d.csv", PBS_ARRAY_INDEX)
s_matrix <- do.call(rbind, s)
write.table(s_matrix, file = predictions_path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

factors_path <- sprintf("factors-%d.csv", PBS_ARRAY_INDEX)
f_matrix <- do.call(rbind, factors) 
write.table(f_matrix, file = factors_path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

# Write configuration variables to a text file
output_file <- "backtesting_hyp.txt"
f <- file(output_file, "w")

writeLines("{", f)
writeLines(paste0("'SEED':'", SEED, "'"), f)
writeLines(paste0("'n_backtest_days_total':'", n_backtest_days_total, "'"), f)
writeLines(paste0("'first_prediction_day':'", first_prediction_day, "'"), f)
writeLines(paste0("'Q':'", Q, "'"), f)
writeLines(paste0("'target_feature':'", target_feature, "'"), f)
writeLines("}", f)

close(f)
