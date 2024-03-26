from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
from sklearn.ensemble import ExtraTreesRegressor
import os
import pandas as pd
import numpy as np
from numpy.random import RandomState
import sys
from sklearn.preprocessing import StandardScaler

def CalMetrics(y_true, y_pred):
    
    r2 = r2_score(y_true=y_true,y_pred=y_pred)
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred) ** 0.5
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    df_y = pd.DataFrame({"y_true":y_true,"y_pred":y_pred})

    pearson_corr = (df_y.corr().iloc[0,1])**2
    return rmse, mae, pearson_corr
def CalMetrics_format3(y_true, y_pred):
    
    r2 = r2_score(y_true=y_true,y_pred=y_pred)
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred) ** 0.5
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    df_y = pd.DataFrame({"y_true":y_true,"y_pred":y_pred})

    pearson_corr = (df_y.corr().iloc[0,1])**2

    return float("{:.3f}".format(rmse)), float("{:.3f}".format(mae)), float("{:.3f}".format(pearson_corr))

def draw_plt(exp, pred, rmse, mae, R2, type):
    #type = "testing"
    #type = "external_testing"

    plt.figure(figsize=(6, 6))
    plt.scatter(exp, pred, c='b', label='Data Points')
    min_val = min(min(exp), min(pred))
    max_val = max(max(exp), max(pred))

    plt.xlim(min_val - (max_val-min_val)*0.2, max_val + (max_val-min_val)*0.2)
    plt.ylim(min_val - (max_val-min_val)*0.2, max_val + (max_val-min_val)*0.2)

    plt.plot([min_val - (max_val-min_val)*0.2, max_val + (max_val-min_val)*0.2], [min_val - (max_val-min_val)*0.2, max_val + (max_val-min_val)*0.2], 'r--', label='y=x Line')

    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)

    plt.xlabel('Exp_k_r',fontsize=14)
    plt.ylabel('ML_pred_k_r',fontsize=14)
    plt.suptitle('Comparison between Exp_k_r and pred_k_r on the {} set'.format(type,type),y=0.01)

    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    plt.text(0.99, 0.13, f'MAE: {mae:.2f}', horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.99, 0.08, f'RMSE: {rmse:.2f}', horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.99, 0.03, f'R²: {R2:.2f}', horizontalalignment='right', verticalalignment='center', transform=plt.gca().transAxes)

    plt.savefig('{}_scatter_plot.png'.format(type), dpi=300, bbox_inches='tight')

standardScaler = StandardScaler()

dataset_train = pd.read_csv("correlated_train.csv",header=None)
dataset_test= pd.read_csv("correlated_test.csv",header=None)
dataset_out= pd.read_csv("correlated external.csv",header=None)
# print(dataset_out)
dataset_train.iloc[:, [-40,-38,-5,-4,-3,-2]] = np.log(dataset_train.iloc[:, [-40,-38,-5,-4,-3,-2]])    
dataset_test.iloc[:, [-40,-38,-5,-4,-3,-2]] = np.log(dataset_test.iloc[:, [-40,-38,-5,-4,-3,-2]])    

basic_num_of_features_including_refra=int(256)

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

emb_fea_out = dataset_out.iloc[:, 0:basic_num_of_features_including_refra]
target_fea_out = dataset_out.iloc[:, basic_num_of_features_including_refra+5] #this is a series
ML_pred_kr_fea_out = dataset_out.iloc[:, basic_num_of_features_including_refra+1:basic_num_of_features_including_refra+5]
QM_fea_out = dataset_out.iloc[:, -45:-1]
solvent_fea_out = dataset_out.iloc[:, -1]


data_train = pd.concat([ML_pred_kr_fea_train.iloc[:,3],QM_fea_train,solvent_fea_train],axis=1)
data_test = pd.concat([ML_pred_kr_fea_test.iloc[:,3],QM_fea_test,solvent_fea_test],axis=1)
data_out = pd.concat([ML_pred_kr_fea_out.iloc[:,3],QM_fea_out,solvent_fea_out],axis=1)

data_train.columns = range(len(data_train.columns))
data_test.columns = range(len(data_test.columns))
data_out.columns = range(len(data_out.columns))



train_X = data_train
train_y = target_fea_train



test_X = data_test
test_y = target_fea_test

out_X = data_out
out_y = target_fea_out




standardScaler.fit(train_X)
X_train_standard = standardScaler.transform(train_X)
X_test_standard = standardScaler.transform(test_X)
X_out_standard = standardScaler.transform(out_X)
numeric_features = train_X.select_dtypes(include=[np.number]).columns
# print(numeric_features)
# print(len(numeric_features))

file = open('./result_collection.txt', 'w')


best_model = XGBRegressor(random_state=42,colsample_bytree=0.6751198707743986,gamma=0.10431438498487772,learning_rate=0.06903265479730043,max_delta_step=0.5027440166274143,max_depth=14,min_child_weight=3,n_estimators=284,reg_alpha=0.029968094540700263,reg_lambda=0.9304652409567173,scale_pos_weight=0.12164983243422732,subsample=0.6502106773952542)

reg_scale = best_model.fit(X_train_standard, train_y)
# # if reg.score(out_X, out_y)> 0.60:
# print("R2 after standard on test",reg.score(X_test_standard, test_y))
# print("R2 after standard on out",reg.score(X_out_standard, out_y))
y_train_pred_scale = reg_scale.predict(X_train_standard)
y_test_pred_scale = reg_scale.predict(X_test_standard)
y_out_pred_scale = reg_scale.predict(X_out_standard)


# print(reg.score(test_X, test_y))
# print(reg.score(out_X, out_y))
print("*"*20, file=file)
print("this is for test_scale", file=file)
print("*"*20, file=file)
print("rmse, mae, squared pearson_corr", file=file)

print(CalMetrics(test_y, y_test_pred_scale), file=file)
print(CalMetrics_format3(test_y, y_test_pred_scale), file=file)

draw_plt(test_y, y_test_pred_scale, CalMetrics(test_y, y_test_pred_scale)[0], CalMetrics(test_y, y_test_pred_scale)[1], CalMetrics(test_y, y_test_pred_scale)[2], "testing")
print("*"*20, file=file)
print("this is for out_scale", file=file)
print("*"*20, file=file)
print("rmse, mae, squared pearson_corr", file=file)

print(CalMetrics(out_y, y_out_pred_scale), file=file)
print(CalMetrics_format3(out_y, y_out_pred_scale), file=file)

draw_plt(out_y, y_out_pred_scale, CalMetrics(out_y, y_out_pred_scale)[0], CalMetrics(out_y, y_out_pred_scale)[1], CalMetrics(out_y, y_out_pred_scale)[2], "external_testing")



print("this is for train_scale", file=file)
print("*"*20, file=file)
print("rmse, mae, squared pearson_corr", file=file)

print(CalMetrics(train_y, y_train_pred_scale), file=file)
print(CalMetrics_format3(train_y, y_train_pred_scale), file=file)

draw_plt(train_y, y_train_pred_scale, CalMetrics(train_y, y_train_pred_scale)[0], CalMetrics(train_y, y_train_pred_scale)[1], CalMetrics(train_y, y_train_pred_scale)[2], "training")
