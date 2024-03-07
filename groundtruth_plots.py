

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename= "ground truth/20240202_DrugCombDB_v20190531_2drug_SynDrugComb_fda_drugbank-id.tsv"

groundtruth_all = pd.read_csv(filename, sep="\t")

filtered_features = groundtruth_all.dropna()
#insgesamt 60 Zeilen mit NA herausgefiltert
filtered_features.to_csv("ground truth/20240202_DrugCombDB_v20190531_2drug_SynDrugComb_fda_drugbank-id_filtered.tsv", sep="\t", index=False)

