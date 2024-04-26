# Predict ICU count data in hospitals throughout UK 

#!/usr/bin/env Rscript
# USAGE: Rscript backtest.R backtesting_config.yaml logMVbedMVC.vts.rda NHSTrustMVCAug120.net.rda

library(igraph)
library("GNAR") 

# Check if the correct number of arguments is provided
if (length(commandArgs(trailingOnly = TRUE)) != 3) {
    stop("USAGE: Rscript backtest.R backtesting_config.yaml ")
}

# Get the config file path from the second argument
config_file <- commandArgs(trailingOnly = TRUE)[1]

# Read the YAML file
yaml_data <- yaml::read_yaml(config_file)

# Access individual variables and assign them to R variables
SEED <- yaml_data$SEED
n_backtest_days_total <- yaml_data$n_backtest_days
first_prediction_day <- yaml_data$first_prediction_day
lookback_window <- yaml_data$lookback_window

logMVbedMVC_vts_rda <- commandArgs(trailingOnly = TRUE)[2]
NHSTrustMVCAug120_net_rda <- commandArgs(trailingOnly = TRUE)[3] 

load(logMVbedMVC_vts_rda) 
load(NHSTrustMVCAug120_net_rda) 

s <- matrix(NA, nrow = n_backtest_days_total, ncol = 140)

for (t in 1:n_backtest_days_total){
    print(t)
    design_matrix <- logMVbedMVC.vts[1:(lookback_window + t - 1),] 
    # X_train_diff <- diff(design_matrix)
    fit_gnar <- GNARfit(vts = design_matrix, 
                    net = NHSTrustMVCAug120.net, 
                    alphaOrder = 1,
                    betaOrder = c(1),
                    globalalpha = FALSE 
                    )    
    predictions <- predict(fit_gnar)  
    # s[t,] <- sum((predictions - logMVbedMVC.vts[t + lookback_window, ])^2) 
    # s[t,] <- sum((predictions - (logMVbedMVC.vts[t + lookback_window, ]-logMVbedMVC.vts[t + 441, ]))^2) 
    # s[t,] <- predictions + design_matrix[(lookback_window + t - 1), , drop=FALSE]
    s[t,] <- predictions 
}


###### SAVE TO FILE ######
predictions_path <- sprintf("predictions.csv")
write.table(s, file = predictions_path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

# Write configuration variables to a text file
output_file <- "backtesting_hyp.txt"
f <- file(output_file, "w")

writeLines("{", f)
writeLines(paste0("'SEED':'", SEED, "'"), f)
writeLines(paste0("'n_backtest_days_total':'", n_backtest_days_total, "'"), f)
writeLines(paste0("'first_prediction_day':'", first_prediction_day, "'"), f)
writeLines("}", f)

close(f)
