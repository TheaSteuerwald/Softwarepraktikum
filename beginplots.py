import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt

#our datatable fpr ctd_cardiovascular
ctd_cardiovascular= pd.read_csv("ctd_cadiovascular_20240223_scores.tsv", sep='\t')


meanspAD = pd.read_csv("ctd_cadiovascular_20240223_scores.tsv",sep='\t', usecols=["meanspAD"])
meanspBD = pd.read_csv("ctd_cadiovascular_20240223_scores.tsv",sep='\t', usecols=["meanspBD"])
meanspAB = pd.read_csv("ctd_cadiovascular_20240223_scores.tsv",sep='\t', usecols=["meanspAB"])

zTDA = pd.read_csv("ctd_cadiovascular_20240223_scores.tsv",sep='\t', usecols=["zTDA"])
zTDB = pd.read_csv("ctd_cadiovascular_20240223_scores.tsv",sep='\t', usecols=["zTDB"])

#filtern in column A und B respectively, sodass jede Target Drug nur 1 mal vorkommt
unique_df = ctd_cardiovascular.drop_duplicates(subset=['drugA'])
unique_df = unique_df.drop_duplicates(subset=['drugB'])

#reset index für alle
unique_df.reset_index(drop=True, inplace=True)
#print(unique_df) #print die "unique" Tabelle

#Vergleich von means zwischen Target A und cardiovascular und
# target B und cardiovascular, sie sind gleich
#Vermutung: da meiste Drug Targets
#sowohl als A als auch als B vorkommen

fig, ax = plt.subplots(1,2)
ax[0].hist(unique_df["meanspAD"],bins= 100,color="pink")
ax[1].hist(unique_df["meanspBD"],bins=100,color="orange")

plt.savefig("meanspAD_meanspBD.png")
plt.close()

#unbereinigte meanspAD und meanspBD
plt.hist(meanspAD,color="pink")
plt.savefig("meanspAD.png")
plt.close()
plt.hist(meanspBD,color="orange")
plt.savefig("meanspBD.png")

#todo woher kommen mean/sd, aus drug target bzw disease genes oder random?
#über cardiovascular genes summierter vergleich von drug target a und cadiovascular
plt.close()
plt.hist(unique_df["zTDA"], bins= np.arange(-1.2,1,0.1))
plt.savefig("hist_zTDA_ber.png")

#über drug target a summierter vergleich von drug target a und cadiovascular
plt.close()
plt.hist(unique_df["zDTA"], bins= np.arange(-1.2,1,0.1))
plt.savefig("hist_zDTA_ber.png")


plt.close()
plt.hist(unique_df["zTDB"], bins= np.arange(-1.2,1,0.1))
plt.savefig("hist_zTDB_ber.png")

plt.close()
plt.hist(unique_df["zDTB"], bins= np.arange(-1.2,1,0.1))
plt.savefig("hist_zDTB_ber.png")

