#Thea Steuerwald, Gesa Röefzaad, Sofya , Lilly Wiesmann

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#insert any of the 8 feature files,example with cardiovascular in primekg/ctd
filename_primekg= "primekg_cardiovascular_20240223_scores.tsv"
filename_ctd= "ctd_cadiovascular_20240223_scores.tsv"

#create a feature dataframe
features_df_primekg = pd.read_csv(filename_primekg, sep='\t')
features_df_ctd = pd.read_csv(filename_ctd, sep='\t')

#first step we want to filter out columns sA,sB,opA,opB because they're empty
columns_to_remove = ['sA', 'sB', 'opA', 'opB']
filtered_features_df_primekg = features_df_primekg.drop(columns=columns_to_remove)
filtered_features_df_ctd = features_df_ctd.drop(columns=columns_to_remove)

#todo : should we save it to a new .tsv ?
#saving the file without the 4 columns
#filtered_features_df.to_csv(f"{filename}_filtered.tsv", sep='\t', index=False)

#comparision of zTDA and zDTA
plt.scatter(filtered_features_df_primekg['zTDA'], filtered_features_df_primekg['zDTA'], marker=".", color="deeppink")
plt.title('Drug A: z-Score Drug-Target vs. z-Score Disease Genes')
plt.xlabel('zTDA')
plt.ylabel('zDTA')
plt.show()
plt.close()

#comparision of zTDB and zDTB shows the same results in the plot

'''
Beispiel-Interpretation für primekg_cardiovascular_20240223_scores.tsv:
-in der Mitte größtes Cluster an zDTA/zTDA scores --> für die meisten zDTA/zTDA
z scores sehr nah am mean 
-generell leichter verschiebung unter 0 bei zDTA also eventuell Korrelation höherer
zTDA --> geringerer zDTA (allerdings nicht wirklich deutlich hier

'''


correlation_AD = filtered_features_df_primekg['sAB'].corr(filtered_features_df_primekg['sAD'])
correlation_BD = filtered_features_df_primekg['sAB'].corr(filtered_features_df_primekg['sBD'])
print(f"correlation_AD: {correlation_AD}")
print(f"correlation_BD: {correlation_BD}")

'''
correlation for sAB und sAD is 0.6522542652536124 and for sAB and sBD 0.6984095234474876
means that with sAB we will be able to predict the separation score sBD better than the seperation 
score sAD 
'''

print(np.mean(filtered_features_df_primekg['meanspAB']))
print(np.mean(filtered_features_df_primekg['meanspAD']))
print(np.mean(filtered_features_df_primekg['meanspBD']))
print("\n")

'''
durchschnittlicher Abstand target drug A zu darget drug B ist etwa 0.5061 und
durchscnittlicher ABstand drug A zum disease gene ist 0.5058, durchschnittlicher Abstand
 drug B zum disease gene ist 0.5043
 --> sehr ähnlicher Abstand zwischen paarweise drugs A,B und drugs zu disease gene
'''

print(np.mean(filtered_features_df_primekg['meanspAB']))
print(np.mean(filtered_features_df_ctd['meanspAB']))

'''
nur als check-up: zwischen den datenbank herrscht keiner bzw ein sehr minimaler unterschied in dem meanspAB
da wir die gleichen drug targets betrachten und das gleiche disease gene
'''

#todo: these are different:why?
print(np.mean(filtered_features_df_primekg['zTDA']))
print(np.mean(filtered_features_df_ctd['zTDA']))



fig, ax = plt.subplots(1,2)
ax[0].hist(filtered_features_df_primekg["meanspAD"],bins= 100,color="deeppink")
ax[1].hist(filtered_features_df_primekg["meanspBD"],bins=100,color="orange")
#plt.show()
plt.savefig("meanspAD_meanspBD.png")
plt.close()

'''
visualization of where the meanspAD and meanspBD are different/the same 
--> similar distribution in general, but sometimes bigger deviations in bins 
'''

fig, px = plt.subplots(1,2)
#über cardiovascular genes summierter vergleich von drug target a und cadiovascular
px[0].hist(filtered_features_df_primekg["zTDA"], bins= 100,color="deeppink")
#über drug target a summierter vergleich von drug target a und cadiovascular
px[1].hist(filtered_features_df_primekg["zDTA"], bins= 100,color="orange")
plt.savefig("zTDA_zDTA.png")
'''
visualization of where the zTDA and zDTA are different/the same 
--> they have many more differences, generally both tend to have more negative values
which means the mean used in both zTDA and zDTA tends to be higher than the distance 
between A and D 
'''
