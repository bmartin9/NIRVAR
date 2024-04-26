# Script to transform August 2022 FRED-MD vintage to make the series stationary. Script taken from https://github.com/cykbennie/fbi
library(fbi)

filepath <- "../../data/raw/FRED-MD/2022-08.csv" 
# filepath <- "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/2022-08.csv"
data <- fredmd(filepath, date_start = as.Date('1960-01-01'), date_end = as.Date('2019-12-01'), transform = TRUE)
N <- ncol(data)

# View the head lines of data
# head(data)

# data_clean <- rm_outliers.fredmd(data)
data_clean <- data
# head(data)

col_na_prop <- apply(is.na(data_clean), 2, mean)
data_select <- data_clean[, (col_na_prop < 0.04)]
data_bal <- na.omit(data_select)
# data_bal <- data_select
X_bal <- data_bal[,2:ncol(data_bal)] 
rownames(X_bal) <- data_bal[,1]

for (i in 1:nrow(X_bal)) {
  if (any(is.na(X_bal[i, ]))) {
    missing_columns <- names(X_bal)[which(is.na(X_bal[i, ]))]
    print(paste("Row", i, "contains missing values in columns:", paste(missing_columns, collapse = ", ")))
  }
}

# View balanced data
# head(X_bal)

# print(matrix(X_bal)[1,])

write.table(X_bal, file = "fred-balanced.csv", sep = ",", row.names = TRUE, col.names = TRUE)