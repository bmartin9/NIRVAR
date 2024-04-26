# Use the MVN package to test whether the sample points are from a MVN distribution 

#!/usr/bin/env Rscript
# USAGE: Rscript compute_statistics.R X_Sample_Covariance.csv hyperparameters.yaml

library("MVN")

# Check if the correct number of arguments is provided
if (length(commandArgs(trailingOnly = TRUE)) < 2) {
    stop("USAGE: Rscript compute_statistics.R X_Sample_Covariance.csv")
}

# Get the config file path from the second argument
X_sample_cov_file <- commandArgs(trailingOnly = TRUE)[1]
config_file <- commandArgs(trailingOnly = TRUE)[2]

# Read the YAML file
yaml_data <- yaml::read_yaml(config_file)

# Access individual variables and assign them to R variables
SEED <- yaml_data$SEED
N <- yaml_data$N1
T <- yaml_data$T
Q <- yaml_data$Q
B <- yaml_data$B
d <- yaml_data$d
sigma <- yaml_data$sigma
N_replicas <- yaml_data$N_replicas
H <- yaml_data$H
spectral_radius <- yaml_data$spectral_radius
n_iter <- yaml_data$n_iter


df <- read.csv(X_sample_cov_file,header=FALSE)

component_to_visualise = 21

subset_indices <- component_to_visualise + seq(0,(N_replicas-1)*N,by=N)

df_component = df[subset_indices,]
# df_component = df[150:300,]

result = mvn(data = df_component,
            mvnTest = "hz",
            scale = TRUE,
            univariateTest = "AD",
            univariatePlot = "histogram",
            multivariatePlot = "qq",
            multivariateOutlierMethod = "none",
            showOutliers = FALSE, 
            showNewData = FALSE)

multi_norm <- result$multivariateNormality
print(multi_norm) 

uni_normality <- result$univariateNormality
print(uni_normality)

descriptives <- result$Descriptives
print(descriptives)

# save multi_norm to csv file for all components 
p_vals <- matrix(nrow = N, ncol = 1)
for (i in 1:N) {
    component_to_visualise <- i

    subset_indices <- component_to_visualise + seq(0, (N_replicas-1)*N, by = N)

    df_component <- df[subset_indices,]

    result <- mvn(data = df_component,
                mvnTest = "hz",
                scale = TRUE,
                univariateTest = "AD",
                univariatePlot = "histogram",
                multivariatePlot = "qq",
                multivariateOutlierMethod = "none",
                showOutliers = FALSE, 
                showNewData = FALSE)
    
    multi_norm <- result$multivariateNormality
    p_vals[i] <- multi_norm[["p value"]]
    
}

write.table(p_vals, file = "hz_pvals.csv", sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

