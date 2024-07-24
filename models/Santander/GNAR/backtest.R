# Predict Number of daily rides using GNAR applied to Santander 

#!/usr/bin/env Rscript
# USAGE: Rscript backtest.R adjacency.csv backtesting_config.yaml <DESIGN_MATRIX>.csv 

library(igraph)
library("GNAR") 

# SCALING FUNCTIONS 
min_max_scaler_matrix <- function(x, range = c(0, 1)) {
  min_vals <- apply(x, 2, min, na.rm = TRUE)
  max_vals <- apply(x, 2, max, na.rm = TRUE)

  scaled_x <- sweep(sweep(x, 2, min_vals, "-"), 2, (max_vals - min_vals), "/")
  scaled_x <- sweep(scaled_x, 2, range[2] - range[1], "*") + range[1]

  return(list(scaled_data = scaled_x, min_vals = min_vals, max_vals = max_vals))
}

inverse_min_max_scaler_matrix <- function(scaled_x, min_vals, max_vals, range = c(0, 1)) {
  # Rescale the data to the original range before min-max scaling
  rescaled_x <- sweep(scaled_x, 2, range[2] - range[1], "*")
  rescaled_x <- sweep(rescaled_x, 2, max_vals - min_vals, "*")

  # Shift the data by the min values
  original_x <- sweep(rescaled_x, 2, min_vals, "+")

  return(original_x)
}


# Check if the correct number of arguments is provided
if (length(commandArgs(trailingOnly = TRUE)) < 3) {
    stop("USAGE: Rscript FNETS-backtest.R backtesting_config.yaml <DESIGN_MATRIX>.csv")
}

# Get the config file path from the second argument
config_file <- commandArgs(trailingOnly = TRUE)[2]

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

# Get a list of days to do backtesting on
days_to_backtest <- 1:n_backtest_days + (n_backtest_days * (PBS_ARRAY_INDEX - 1)) - 1 

# Get the design matrix file path from the second argument
design_matrix_file <- commandArgs(trailingOnly = TRUE)[3]

# Read the design matrix
Xs <- read.csv2(design_matrix_file,sep=",",dec = ".",header=FALSE)
Xs[] <- lapply(Xs, as.numeric)

# Get dimensions
T <- nrow(Xs)
N_times_Q <- ncol(Xs)

N <- N_times_Q / Q

# Get the adjacency matrix file path from the first argument    
adjacency_file <- commandArgs(trailingOnly = TRUE)[1]   

# Read the adjacency matrix 
adjacency_matrix <- read.csv(adjacency_file, header = FALSE) 
adjacency_matrix <- matrix(as.integer(data.matrix(adjacency_matrix)), nrow = nrow(adjacency_matrix))

# Check that the adjacency matrix is symmetric
is_symmetric <- function(matrix) {
#   return(identical(matrix, t(matrix)))
  return(all(matrix == t(matrix)))
}
# is_symmetric(adjacency_matrix)

sic_net <- graph_from_adjacency_matrix(adjacency_matrix, 'undirected')
gnar_net <- igraphtoGNAR(sic_net)

s <- matrix(NA, nrow = n_backtest_days, ncol = N)


for (index in 1:n_backtest_days){
    t <- days_to_backtest[[index]] 
    todays_date <- first_prediction_day + t
    print(index)
    furthest_lookback_day <- todays_date - lookback_window +1
    X_train <- Xs[furthest_lookback_day:(todays_date),] 
    X_train <- data.matrix(X_train)
    X_train_diff <- diff(X_train, lag = 1)
    # scaled_list <- min_max_scaler_matrix(X_train)
    # X_train_scaled <- scaled_list$scaled_data
    # X_train_min <- scaled_list$min_val
    # X_train_max <- scaled_list$max_val 
    # col_means <- colMeans(X_train_scaled)
    # X_train_scaled <- sweep(X_train_scaled,2,col_means,FUN = "-")
    fit_gnar <- GNARfit(vts = X_train_diff, 
                    net = gnar_net, 
                    alphaOrder = 1,
                    betaOrder = c(1),
                    globalalpha = TRUE
                    )
    # print(object.size(fit_gnar)) 
    predictions <- predict(fit_gnar, n.ahead = 1)
    # predictions_scaled <- predict(fit_gnar, n.ahead = 1)
    # predictions_scaled <- sweep(predictions_scaled,2,col_means,FUN = "+") 
    # predictions <- inverse_min_max_scaler_matrix(predictions_scaled,min_vals = X_train_min,max_vals = X_train_max) 
    # Check if all elements in the matrix are 0
    if(any(predictions == 0)) {
      print("The predictions matrix contains all 0s.") 
      print(fit_fnets) 
    }  
    # s[index,] <- predictions + X_train[nrow(X_train),] 
    s[index,] <- predictions
    rm(fit_gnar)
}

###### SAVE TO FILE ######
predictions_path <- sprintf("predictions-%d.csv", PBS_ARRAY_INDEX)
write.table(s, file = predictions_path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

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
