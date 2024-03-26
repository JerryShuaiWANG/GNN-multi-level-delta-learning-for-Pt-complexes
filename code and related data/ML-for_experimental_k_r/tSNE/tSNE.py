from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
import xgboost as xgb
import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import os
# import hyperopt
import pandas as pd
import numpy as np
from numpy.random import RandomState
import sys
# import joblib
import matplotlib.pyplot as plt
# from PIL import Image
# import io
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D


coor_list = [
[2,2,3,3], #cooridination type: ONNO 
[1,2,2,3], #cooridination type: ONCN
[1,2,2,4], #cooridination type: CNNCl 
[1,1,2,2], #cooridination type: NCCN 
[1,1,3,3], #cooridination type: OCCO 
[1,1,1,2], #cooridination type: CCCN 
[1,1,1,1],
[2,2,2,2],
[1,2,2,2],
[1,2,3,3],
[1,1,2,1],
[2,1,2,4],
[1,2,1,2],
[1,2,1,1],
[1,2,2,1],
[1,2,3,2],
[1,4,2,2],
    ]
basic_num_of_features_including_refra = 256 

dataset_train = pd.read_csv(r"correlated_train.csv", header= None)
dataset_test = pd.read_csv(r"correlated_test.csv", header= None)

dataset_train.iloc[:, [-40,-38,-5,-4,-3,-2]] = np.log(dataset_train.iloc[:, [-40,-38,-5,-4,-3,-2]])    
dataset_test.iloc[:, [-40,-38,-5,-4,-3,-2]] = np.log(dataset_test.iloc[:, [-40,-38,-5,-4,-3,-2]])    

emb_fea_train = dataset_train.iloc[:, 0:basic_num_of_features_including_refra]
target_fea_train = dataset_train.iloc[:, basic_num_of_features_including_refra+5] #this is a series
ML_pred_kr_fea_train = dataset_train.iloc[:, basic_num_of_features_including_refra+1:basic_num_of_features_including_refra+5]
QM_fea_train = dataset_train.iloc[:, -45:-1]
solvent_fea_train = dataset_train.iloc[:, -1]

emb_fea_test = dataset_test.iloc[:, 0:basic_num_of_features_including_refra]
target_fea_test = dataset_test.iloc[:, basic_num_of_features_including_refra+5] #this is a series
ML_pred_kr_fea_test = dataset_test.iloc[:, basic_num_of_features_including_refra+1:basic_num_of_features_including_refra+5]
QM_fea_test = dataset_test.iloc[:, -45:-1]
solvent_fea_test = dataset_test.iloc[:, -1]

data_train = pd.concat([ML_pred_kr_fea_train.iloc[:,3],QM_fea_train,solvent_fea_train],axis=1)
data_test = pd.concat([ML_pred_kr_fea_test.iloc[:,3],QM_fea_test,solvent_fea_test],axis=1)

your_feature_names = ["ML_pred_kr",	"HOMO",	"LUMO",	"dipolemoment",	"H_T1_S1",	"H_T1_S0",	"f",	"emission_energy",	"density_Pt",	"CT_S1",	"R_EH _S1a",	"LAMBDA_S1",	"R_EH _S1b",	"CT_S2",	"R_EH _S2a",	"LAMBDA_S2",	"R_EH _S2b",	"CT_S3",	"R_EH _S3a",	"LAMBDA_S3",	"R_EH _S3b",	"CT_T1",	"R_EH _T1a",	"LAMBDA_T1",	"R_EH _T1b",	"CT_T2",	"R_EH _T2a",	"LAMBDA_T2",	"R_EH _T2b",	"CT_T3",	"R_EH _T3a",	"LAMBDA_T3",	"R_EH _T3b",	"coor_bond_type (1)",	"coor_bond_type (2)",	"coor_bond_type (3)",	"coor_bond_type (4)",	"coor_bond_length (1)",	"coor_bond_length (2)",	"coor_bond_length (3)",	"coor_bond_length (4)",	"density_coor (1)",	"density_coor (2)",	"density_coor (3)",	"density_coor (4)",	"refractive index"]

train_X = data_train
train_X.columns=your_feature_names

test_X = data_test
test_X.columns=your_feature_names

target_fea_train = pd.DataFrame(target_fea_train)
target_fea_test = pd.DataFrame(target_fea_test)

train_y = target_fea_train
test_y = target_fea_test

standardScaler = StandardScaler()
standardScaler.fit(train_X)
X_train_standard = standardScaler.transform(train_X)
X_train_standard = pd.DataFrame(X_train_standard, columns=your_feature_names)

X_test_standard = standardScaler.transform(test_X)
X_test_standard = pd.DataFrame(X_test_standard, columns=your_feature_names)

train_data = X_train_standard 
test_data = X_test_standard  

merged_data = pd.concat([train_data, test_data], axis=0)


train_coor_X, test_coor_X = QM_fea_train.iloc[:,-12:-8],QM_fea_test.iloc[:,-12:-8]
coor_X_all = pd.concat([train_coor_X,test_coor_X],axis=0)


# print(coor_X_all)
coor_X_all_reindex = coor_X_all.reset_index(drop=True)
# print(coor_X_all_reindex)
# print(type(coor_X_all_reindex))

coor_X_all_reindex_sorted = coor_X_all_reindex.apply(lambda row: row.tolist(), axis=1)  

print(coor_X_all_reindex_sorted)
print(type(coor_X_all_reindex_sorted))

print(merged_data)
merged_data_reindex = merged_data.reset_index(drop=True)
print(merged_data_reindex)


features = merged_data_reindex

color_list = [
            'green',
            'yellow',
            'blue',
            'red',
            'orange',
            'purple',
           'pink',
            'gray',
           'cyan',
           'magenta',
            'brown',
            'black',
            'teal',
            'navy',
            'olive',
            'gold',
            'silver',
            ]

tsne = TSNE(n_components=2, random_state=8)

tsne_data = tsne.fit_transform(features)


for i in range(len(tsne_data)):
    plt.scatter(tsne_data[i, 0], tsne_data[i, 1], c=color_list[coor_list.index(coor_X_all_reindex_sorted[i])])

plt.title('tSNE visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()

plt.savefig("tSNE_standard_tr_te_coor_rs_{}.png".format(8))

plt.clf()
