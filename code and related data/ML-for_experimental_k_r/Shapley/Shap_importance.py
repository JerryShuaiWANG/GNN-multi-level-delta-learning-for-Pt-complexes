import shap
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import os
import pandas as pd
import numpy as np
from numpy.random import RandomState
import sys
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import io
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler


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
X_test_standard = standardScaler.transform(test_X)


knn_model = XGBRegressor(random_state=42,colsample_bytree=0.6751198707743986,gamma=0.10431438498487772,learning_rate=0.06903265479730043,max_delta_step=0.5027440166274143,max_depth=14,min_child_weight=3,n_estimators=284,reg_alpha=0.029968094540700263,reg_lambda=0.9304652409567173,scale_pos_weight=0.12164983243422732,subsample=0.6502106773952542)  


y_train = train_y
X_train = X_train_standard
model = knn_model.fit(X_train, y_train)
explainer = shap.TreeExplainer(model,X_train)

test_X = X_test_standard
shap_values = explainer.shap_values(X_train)
# print(shap_values)
your_feature_names = ["ML_pred_kr",	"HOMO",	"LUMO",	"dipolemoment",	"H_T1_S1",	"H_T1_S0",	"f",	"emission_energy",	"density_Pt",	"CT_S1",	"R_EH _S1a",	"LAMBDA_S1",	"R_EH _S1b",	"CT_S2",	"R_EH _S2a",	"LAMBDA_S2",	"R_EH _S2b",	"CT_S3",	"R_EH _S3a",	"LAMBDA_S3",	"R_EH _S3b",	"CT_T1",	"R_EH _T1a",	"LAMBDA_T1",	"R_EH _T1b",	"CT_T2",	"R_EH _T2a",	"LAMBDA_T2",	"R_EH _T2b",	"CT_T3",	"R_EH _T3a",	"LAMBDA_T3",	"R_EH _T3b",	"coor_bond_type (1)",	"coor_bond_type (2)",	"coor_bond_type (3)",	"coor_bond_type (4)",	"coor_bond_length (1)",	"coor_bond_length (2)",	"coor_bond_length (3)",	"coor_bond_length (4)",	"density_coor (1)",	"density_coor (2)",	"density_coor (3)",	"density_coor (4)",	"refractive index"]

shap.summary_plot(shap_values, features=X_train, feature_names=your_feature_names,plot_size=(10, 20),max_display=len(your_feature_names),show=False)  # 替换成你的特征名称

plt.savefig('summary_plot.png', dpi=300, bbox_inches='tight')
