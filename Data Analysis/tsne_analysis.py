#tsne

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#for LOO
from sklearn.model_selection import LeaveOneOut
#for TSNE
from sklearn.manifold import TSNE

#############################################################
# Step 1: importing the dataset and creating useful objects #
#############################################################

# Import the Excel dataset into a pandas dataframe, df
df_1 = pd.read_csv("517_halfhourly_cleaned.csv");
df_2 = pd.read_csv("709_halfhourly_cleaned.csv");
Sites = ['Blk 517 W Coast Rd', 'Blk 709 Clementi W St 2']
df_1['Site'] = Sites[0];
df_2['Site'] = Sites[1];
df = pd.concat([df_1, df_2]).reset_index(drop=True);
# Look at df and extract the variables. Make sure that you don't accidentally
# include labels as variables.
variables = list(df.columns.values)
variables = variables[3:7]

# Get the unstandardised data
x_unstd = df.loc[:, variables].values

# The samples are pre-grouped based on Site. Make these the targets, y.
y = df.loc[:,["Site"]].values;

# Standardise the dataset, x.
x = StandardScaler().fit_transform(x_unstd)

# 3. Evaluate KL divergence at various perplexities

#perplexity = np.arange(5, 100, 3)
#perplexity = [5]
divergence = []

#for i in perplexity:
#    model = TSNE(n_components=3, init="pca", perplexity=i)
#    reduced = model.fit_transform(x)
#    divergence.append(model.kl_divergence_)

#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1)
#ax.set_xlabel('Divergence', fontsize = 15)
#ax.set_ylabel('Perplexity', fontsize = 15)
#plt.plot(perplexity,divergence,'ko-')
#plt.show()

perp = 95
tsne = TSNE(n_components=2,perplexity=perp)
x_tSNE = tsne.fit_transform(x)

tSNE_Df2 = pd.DataFrame(data = x_tSNE, columns = ['tSNE1', 'tSNE2'])
tSNE_Df = pd.concat([tSNE_Df2, df[['Site']]], axis = 1)

# 5. Plot it

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('tSNE1', fontsize = 15)
ax.set_ylabel('tSNE2', fontsize = 15)
ax.set_title('tSNE perplexity '+str(perp), fontsize = 20)
targets = ['Blk 517 W Coast Rd', 'Blk 709 Clementi W St 2']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = tSNE_Df['Site'] == target
    ax.scatter(tSNE_Df.loc[indicesToKeep, 'tSNE1']
               , tSNE_Df.loc[indicesToKeep, 'tSNE2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
#plt.show()
plt.savefig('tsne_analysis.png')

output_csv = "tsne_results.csv"
tSNE_Df.to_csv(output_csv, index=False)
print(f"t-SNE results saved to {output_csv}")
