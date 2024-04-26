# Script to run NIRVAR on FRED data for multiple values of the embedding dimension d

#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <DESIGN_MATRIX.csv> <backtesting_config.yaml>"
    exit 1
fi

# Assign the command line arguments to variables
DESIGN_MATRIX="$1"
backtesting_config="$2"

# Define the initial value of d and the increment (4y)
start_value=1
increment=1
y=70  # You can adjust the value of y as needed

# Calculate the end value of d based on the formula 12 + 4y
end_value=$((start_value + increment * y))

echo "x,mse" > overall_mse_d.csv

# Iterate over the range of d values
for ((d = start_value; d <= end_value; d += increment)); do
    echo "Running my_script.py with d = $d"
    python ./NIRVAR/backtest.py "$DESIGN_MATRIX" "$backtesting_config" True $d 
    mse=$(python overall-mse.py "$DESIGN_MATRIX" ./predictions-1.csv)
    rm predictions-1.csv
    # Append the result to the CSV file
    echo "$d,$mse" >> overall_mse_d.csv 
done
