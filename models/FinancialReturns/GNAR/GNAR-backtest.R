# Backtest GNAR model when the observed network is defined using SIC clusters.

#!/usr/bin/env Rscript
# USAGE: Rscript GNAR-backtest.R adjacency.csv backtesting_config.yaml <DESIGN_MATRIX>.csv 

library("GNAR") 
library(yaml)
library(igraph) 

# Check if the correct number of arguments is provided
if (length(commandArgs(trailingOnly = TRUE)) < 3) {
    stop("USAGE: Rscript GNAR-backtest.R adjacency.csv backtesting_config.yaml <DESIGN_MATRIX>.csv")
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
NUM_ARRAY_INDICES = 4275

# Re-define n_backtest_days to be total number of backtesting days divided by the number of array indices 
n_backtest_days <- as.integer(n_backtest_days_total/NUM_ARRAY_INDICES) 

# Get a list of days to do backtesting on
days_to_backtest <- 1:n_backtest_days + (n_backtest_days * (PBS_ARRAY_INDEX - 1)) - 1 

# Get the adjacency matrix file path from the first argument    
adjacency_file <- commandArgs(trailingOnly = TRUE)[1]   

# Read the adjacency matrix 
adjacency_matrix <- read.csv(adjacency_file, header = FALSE) 
adjacency_matrix <- data.matrix(adjacency_matrix)


sic_net <- graph_from_adjacency_matrix(adjacency_matrix, 'undirected')
gnar_net <- igraphtoGNAR(sic_net)

# Get the design matrix file path from the third argument
design_matrix_file <- commandArgs(trailingOnly = TRUE)[3]

# Read the design matrix
Xs <- read.csv2(design_matrix_file,sep=",",dec = ".",header=FALSE)

# Get dimensions
T <- nrow(Xs)
N_times_Q <- ncol(Xs)

N <- N_times_Q / Q

###### SPECIFY GNAR MODEL PARAMETERS ######
alpha_order = 1
beta_order = c(1)
neighbours_error = TRUE 

s <- list()

for (index in 1:n_backtest_days){
    t <- days_to_backtest[[index]] 
    todays_date <- first_prediction_day + t 
    furthest_lookback_day <- todays_date - lookback_window +1
    pvCLCL_X <- Xs[furthest_lookback_day:(todays_date),(N+1):(2*N)] 
    print(Xs[todays_date+1,N+1])
    pvCLCL_X <- data.matrix(pvCLCL_X)
    print(dim(pvCLCL_X))
    fit_gnar <- GNARfit(vts = pvCLCL_X, 
                    net = gnar_net, 
                    alphaOrder = alpha_order,
                    betaOrder = beta_order,
                    globalalpha = FALSE
                    )
    print(object.size(fit_gnar)) 
    predictions <- predict(fit_gnar, n.ahead = 1) 
    s[[index]] <- predictions[1:N] 
    rm(fit_gnar) 
}

###### SAVE TO FILE ######
predictions_path <- sprintf("predictions-%d.csv", PBS_ARRAY_INDEX)
s_matrix <- do.call(rbind, s)
write.table(s_matrix, file = predictions_path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

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









