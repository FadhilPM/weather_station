import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#for LOO
from sklearn.model_selection import LeaveOneOut


#############################################################
# Step 1: importing the dataset and creating useful objects #
#############################################################

# Import the Excel dataset into a pandas dataframe, df
df_1 = pd.read_csv("517_cleaned.csv");
df_2 = pd.read_csv("709_cleaned.csv");
Sites = ['Blk 517 W Coast Rd', 'Blk 709 Clementi W St 2']
df_1['Site'], df_2['Site'] = Sites;
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

#############################################
# Step 2: Leave-one-out validation
#############################################
def PRESS_calc(x, PCs):
    loo = LeaveOneOut();
    #the final predicted matrix should be the same size as the original x matrix?
    cross_validated_matrix = np.zeros(x.shape);
    for rest,leftout in loo.split(x): #This method provides the index of the stuff left out and the indices of the stuff left in
        rest_indices = rest;
        leftout_index = leftout;
        rest_data = x[rest_indices];
        leftout_data = x[leftout_index];
        #we will now carry out PCA
        pca = PCA(n_components = PCs)
        scores_matrix = pca.fit_transform(rest_data) #X_1 = T_1 P_1 + E, where the _1 means that the first sample has been removed

        #Above is the PCA scores matrix that has the rest of the data. We will now use this to predict the value of the data that was left out.
        #The scores of the sample that was removed can be predicted from the sample readings
        #and the LOO loadings
        #Loadings are the P matrix
        #The score estimate is found by multiplying the sample
        #and the transpose of the loadings matrix
        P = pca.components_; # This is P_1
        P_T = P.T; # This is P_1T
        test = leftout_data @ P.T #this should be equal to ^t_1 = x_1 P_1T
        x1 = test @ P; #this is cross-validated predicted sample, which should be A,cv ^x1 = A^t_1 AP_1
        cross_validated_matrix[leftout_index] = x1;
    difference_matrix = cross_validated_matrix - x;
    PRESS = np.sum(difference_matrix ** 2);
    return PRESS

def RMS_calc(x, PCs):
    pca = PCA(n_components = PCs);
    scores_matrix = pca.fit_transform(x);
    P = pca.components_
    reconstructed_matrix = scores_matrix @ P;
    E = x - reconstructed_matrix;
    RSS = np.sum(E**2);
    #print(reconstructed_matrix);
    return RSS;

PRESS_list = []
RMS_list = []
for i in range(1, len(variables)+1):
    PRESS_list.append(PRESS_calc(x, i));
    RMS_list.append(RMS_calc(x, i));

PCs = len(variables);
for j in range(0, len(variables)-1):
    if(RMS_list[j] < PRESS_list[j+1]):
        PCs = j;

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(variables) + 1), RMS_list, label='RMS', marker='o')
plt.plot(range(1, len(variables) + 1), PRESS_list, label='PRESS', marker='s')

plt.xlabel('Number of Principal Components', fontsize=12)
plt.ylabel('Error Values', fontsize=12)
plt.title('RMS and PRESS vs n(Principal Components)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig('RMS_PRESS_plot.png');
print(PCs);

#############################################
# Step 3: Perform PCA
#############################################
PClist = [];
for i in range(PCs):
    PClist.append(f'PC{i+1}')
pca = PCA(n_components = PCs);

# Get the scores matrix
scores_matrix = pca.fit_transform(x)
scores = pd.DataFrame(data = scores_matrix, columns = PClist) # Converts the result into a df

# Also get loadings for later plotting
loadings = pd.DataFrame(pca.components_.T, columns=PClist, index=variables)
loadings.to_csv('Loadings.csv')
loadingsT = np.transpose(loadings)

# Concatenate the scores matrix, scores, with the Site labelling
finalDf = pd.concat([scores, df[['Site']]], axis = 1);

#################################################################
# Step 4: Make the biplot, using Site to colour the datapoints #
#################################################################
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Scores plot, 2 component PCA', fontsize = 20)

colors = ['r','b'];

for Site, color in zip(Sites, colors):
    indicesToKeep = finalDf['Site'] == Site;
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 10)
ax.legend(Sites)
ax.grid()
loading_values = loadings.loc[:].values
scale = 5.0
for i in range(np.size(variables)):
    lx, ly = (loading_values[i,:]*scale)[:2]
    xx, yy = np.linspace(0,lx,101), np.linspace(0,ly,101)
    plt.text(lx, ly, loadings.index.values[i])
    plt.plot(xx, yy, linewidth = 1, color='black')

plt.xticks(fontsize = 15);
plt.yticks(fontsize = 15);
plt.savefig('biplot.png')

###############################################################
# Step 4: Calculate and plot the scores eigenvalue of each PC #
###############################################################
scores_eigen = pca.explained_variance_ratio_; #find the right method here!
np.savetxt("eigenscores.csv", scores_eigen, delimiter=",")
plt.figure(figsize=(10,5))
plt.bar(range(1, scores_eigen.size + 1), scores_eigen);
plt.xlabel("Principal component")
plt.ylabel("Normalised eigenvalue")
plt.title("Eigenvalues of principal components")
plt.xticks(range(1, scores_eigen.size + 1), labels = PClist);
plt.savefig('eigenvalues.png')

























