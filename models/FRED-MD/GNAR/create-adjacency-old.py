""" 
Create adjacency matrix for use in GNAR model.
"""

#!/usr/bin/env python
# USAGE: python create-adjacency.py fred-balanced.csv

import numpy as np 
import sys 
import csv

group1 = [
    "RPI",
    "W875RX1",
    "INDPRO",
    "IPFPNSS",
    "IPFINAL",
    "IPCONGD",
    "IPDCONGD",
    "IPNCONGD",
    "IPBUSEQ",
    "IPMAT",
    "IPDMAT",
    "IPNMAT",
    "IPMANSICS",
    "IPB51222s",
    "IPFUELS",
    "NAPMPI",
    "CUMFNS"
]

group2 = [
    "HWI",
    "HWIURATIO",
    "CLF16OV",
    "CE16OV",
    "UNRATE",
    "UEMPMEAN",
    "UEMPLT5",
    "UEMP5TO14",
    "UEMP15OV",
    "UEMP15T26",
    "UEMP27OV",
    "CLAIMSx",
    "PAYEMS",
    "USGOOD",
    "CES1021000001",
    "USCONS",
    "MANEMP",
    "DMANEMP",
    "NDMANEMP",
    "SRVPRD",
    "USTPU",
    "USWTRADE",
    "USTRADE",
    "USFIRE",
    "USGOVT",
    "CES0600000007",
    "AWOTMAN",
    "AWHMAN",
    "NAPMEI",
    "CES0600000008",
    "CES2000000008",
    "CES3000000008"
]

group3 = [
    "HOUST",
    "HOUSTNE",
    "HOUSTMW",
    "HOUSTS",
    "HOUSTW",
    "PERMIT",
    "PERMITNE",
    "PERMITMW",
    "PERMITS",
    "PERMITW"
]

group4 = [
    "DPCERA3M086SBEA",
    "CMRMTSPLx",
    "RETAILx",
    "NAPM",
    "NAPMNOI",
    "NAPMSDI",
    "NAPMII",
    "ACOGNO",
    "AMDMNOx",
    "ANDENOx",
    "AMDMUOx",
    "BUSINVx",
    "ISRATIOx",
    "UMCSENTx"
]

group5 = [
    "M1SL",
    "M2SL",
    "M2REAL",
    "AMBSL",
    "TOTRESNS",
    "NONBORRES",
    "BUSLOANS",
    "REALLN",
    "NONREVSL",
    "CONSPI",
    "MZMSL",
    "DTCOLNVHFNM",
    "DTCTHFNM",
    "INVEST"
]


group6 = [
    "FEDFUNDS",
    "CP3Mx",
    "TB3MS",
    "TB6MS",
    "GS1",
    "GS5",
    "GS10",
    "AAA",
    "BAA",
    "COMPAPFFx",
    "TB3SMFFM",
    "TB6SMFFM",
    "T1YFFM",
    "T5YFFM",
    "T10YFFM",
    "AAAFFM",
    "BAAFFM",
    "TWEXMMTH",
    "EXSZUSx",
    "EXJPUSx",
    "EXUSUKx",
    "EXCAUSx"
]

group7 = [
    "PPIFGS",
    "PPIFCG",
    "PPIITM",
    "PPICRM",
    "OILPRICEx",
    "PPICMM",
    "NAPMPRI",
    "CPIAUCSL",
    "CPIAPPSL",
    "CPITRNSL",
    "CPIMEDSL",
    "CUSR0000SAC",
    "CUUR0000SAD",
    "CUSR0000SAS",
    "CPIULFSL",
    "CUUR0000SA0L2",
    "CUSR0000SA0L5",
    "PCEPI",
    "DDURRG3M086SBEA",
    "DNDGRG3M086SBEA",
    "DSERRG3M086SBEA"
]

group8 = [
    "S&P 500",
    "S&P: indust",
    "S&P div yield",
    "S&P PE ratio"
]

combined_groups = group1 + group2 + group3 + group4 + group5 + group6 + group7 + group8

print("Length of combined_groups:", len(combined_groups))   

# Path to your CSV file
file_path = sys.argv[1] 

# List to store the header
header = []

# Reading the CSV file
with open(file_path, 'r') as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)

    # Read the header
    header = next(csvreader)

print("Header of the CSV file:", len(header))


not_in_header = [element for element in combined_groups if element not in header]
# print("Elements of combined_groups not in header:", not_in_header)


not_in_combined_groups = [element for element in header if element not in combined_groups]
print("Elements of header not in combined_groups:", not_in_combined_groups)

print(header) 

print(combined_groups)



