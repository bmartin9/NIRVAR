# backtest FNETS on Santander dataset 

#!/usr/bin/env Rscript
# USAGE: Rscript backtest.R  backtesting_config.yaml <DESIGN_MATRIX>.csv 

library("fnets") 

# SCALING FUNCTIONS 
min_max_scaler_matrix <- function(x, range = c(-1, 1)) {
  min_vals <- apply(x, 2, min, na.rm = TRUE)
  max_vals <- apply(x, 2, max, na.rm = TRUE)

  scaled_x <- sweep(sweep(x, 2, min_vals, "-"), 2, (max_vals - min_vals), "/")
  scaled_x <- sweep(scaled_x, 2, range[2] - range[1], "*") + range[1]

  return(list(scaled_data = scaled_x, min_vals = min_vals, max_vals = max_vals))
}

inverse_min_max_scaler_matrix <- function(scaled_x, min_vals, max_vals, range = c(-1, 1)) {
  # Rescale the data to the original range before min-max scaling
  scaled_x <- scaled_x - range[1]
  rescaled_x <- sweep(scaled_x, 2, range[2] - range[1], "/") 
  rescaled_x <- sweep(rescaled_x, 2, max_vals - min_vals, "*")

  # Shift the data by the min values
  original_x <- sweep(rescaled_x, 2, min_vals, "+")

  return(original_x)
}

# Check if the correct number of arguments is provided
if (length(commandArgs(trailingOnly = TRUE)) < 2) {
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
FNETS_restricted <- yaml_data$FNETS_restricted

###### ENVIRONMENT VARIABLES ###### 
# PBS_ARRAY_INDEX <- as.numeric(Sys.getenv("PBS_ARRAY_INDEX"))
# NUM_ARRAY_INDICES <- as.numeric(Sys.getenv("NUM_ARRAY_INDICES")) 
PBS_ARRAY_INDEX = 1
NUM_ARRAY_INDICES = 1

# Re-define n_backtest_days to be total number of backtesting days divided by the number of array indices 
n_backtest_days <- as.integer(n_backtest_days_total/NUM_ARRAY_INDICES) 
print(n_backtest_days)

# Get a list of days to do backtesting on
days_to_backtest <- 1:n_backtest_days + (n_backtest_days * (PBS_ARRAY_INDEX - 1)) - 1 

# Get the design matrix file path from the second argument
design_matrix_file <- commandArgs(trailingOnly = TRUE)[2]

# Read the design matrix
Xs <- read.csv2(design_matrix_file,sep=",",dec = ".",header=FALSE)
Xs[] <- lapply(Xs, as.numeric)

# Get dimensions
T <- nrow(Xs)
N_times_Q <- ncol(Xs)

N <- N_times_Q / Q

s <- matrix(NA, nrow = n_backtest_days, ncol = N)

factors <- matrix(NA, nrow = n_backtest_days, ncol = 1)

for (index in 1:n_backtest_days){
    t <- days_to_backtest[[index]] 
    todays_date <- first_prediction_day + t 
    print(todays_date)
    furthest_lookback_day <- todays_date - lookback_window +1
    X_train <- Xs[furthest_lookback_day:(todays_date),] 
    X_train <- data.matrix(X_train)
    # print(X_train[todays_date,1])
    X_train_diff <- diff(X_train, lag = 1)

    scaled_list <- min_max_scaler_matrix(X_train)
    X_train_scaled <- scaled_list$scaled_data
    X_train_min <- scaled_list$min_val
    X_train_max <- scaled_list$max_val 
    col_means <- colMeans(X_train_scaled)
    X_train_scaled <- sweep(X_train_scaled,2,col_means,FUN = "-") 

    fit_fnets <- fnets(x=X_train_scaled,
                    center = FALSE,
                    fm.restricted = FNETS_restricted,
                    q = "ic",
                    var.order = 1,
                    var.method = "lasso",
                    do.threshold = FALSE,
                    do.lrpc = FALSE,
                    ) 
    print(fit_fnets$q)
    print(object.size(fit_fnets)) 
    factors[index,1] <- fit_fnets$q 
    pr <- predict(fit_fnets, n.ahead = 1,fc.restricted = FNETS_restricted)
    predictions_matrix <- pr$forecast  
    # Check if all elements in the matrix are 0
    if(any(predictions_matrix == 0)) {
      print("The predictions matrix contains all 0s.") 
      print(fit_fnets) 
    }  
    predictions_matrix <- sweep(predictions_matrix,2,col_means,FUN = "+")
    predictions_matrix <- inverse_min_max_scaler_matrix(predictions_matrix,min_vals = X_train_min,max_vals = X_train_max) 

    s[index,] <- predictions_matrix 
    # s[index,] <- predictions_matrix + X_train[nrow(X_train),] 
    rm(fit_fnets)
}

###### SAVE TO FILE ######
predictions_path <- sprintf("predictions-%d.csv", PBS_ARRAY_INDEX)
write.table(s, file = predictions_path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

factors_path <- sprintf("factors-%d.csv", PBS_ARRAY_INDEX)
write.table(factors, file = factors_path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

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