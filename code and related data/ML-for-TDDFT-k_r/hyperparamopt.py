#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : QsarTrain.py
@Description  : 
@Time         : 2024/03/13 
@Author       : Shuai Wang, Shengde Zhang
@Version      : 2.0
'''

import os
import hyperopt
import pandas as pd
import numpy as np
from numpy.random import RandomState
import sys

from QsarUtils import *
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from hyperopt import hp
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
# from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from ks_sampling import spxy
from ks_sampling import kennardstone
from find_the_best_and_write_ import cal_perf

def MainRegression(in_file_path, saved_dir, feature_selector_list, select_des_num_list, test_file_path=None, model_list=("RF",), 
         search_max_evals=250, des_type=("all",), smi_column="Smiles", name_column="Name", label_column="", sep=",", group_column="", k=10,
         test_size=0.2,  kfold_type="stratified", random_state=0, search_metric="val_RMSE", greater_is_better=False, target_file=str(), basic_num_of_features_including_refra = 51, target="lambda", split_method="KS_all"):

    dataset = pd.read_csv(target_file, header= None)

    dataset.iloc[:, basic_num_of_features_including_refra] = np.log10(dataset.iloc[:, basic_num_of_features_including_refra])    


    if split_method in ['spxy']:
        if target in ['Calc_Kr']:
            X = dataset.iloc[:,0:basic_num_of_features_including_refra].join(dataset.iloc[:, -3])#only refra
            y = dataset.iloc[:,basic_num_of_features_including_refra]

        X = X.to_numpy()
        y = y.to_numpy()
        X_train_index, X_test_index = spxy(X, y, test_size = 0.2)



    train_after_spxy = dataset.iloc[X_train_index,:]
    test_after_spxy = dataset.iloc[X_test_index,:]

    if target in ['Calc_Kr']:
        train_X, test_X, train_y, test_y = train_after_spxy.iloc[:,0:basic_num_of_features_including_refra].join(train_after_spxy.iloc[:, -3]),test_after_spxy.iloc[:,0:basic_num_of_features_including_refra].join(test_after_spxy.iloc[:, -3]),train_after_spxy.iloc[:, basic_num_of_features_including_refra],test_after_spxy.iloc[:, basic_num_of_features_including_refra]

    Xy_out_file = os.path.join(saved_dir,"a_Xy_train_test.txt")
    writefile = open(Xy_out_file, 'w')
    print(target_file, basic_num_of_features_including_refra, target, split_method, file=writefile)
    print("train_X",train_X,file=writefile)
    print("train_y",train_y,file=writefile)
    print("test_X",test_X,file=writefile)
    print("test_y",test_y,file=writefile)
    writefile.close()
    train_X = train_X.to_numpy()
    train_y = train_y.to_numpy()
    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy()


    model = RegressionModel(random_state=random_state)
    model.LoadData(train_X, train_y, test_X, test_y)
    model.ScaleFeature(saved_dir=saved_dir,saved_file_note="_".join(des_type))
    model.KFoldSplit(k=k, kfold_type=kfold_type)

    def Search(params):
        nonlocal model
        nonlocal estimator
        nonlocal search_metric
        nonlocal greater_is_better
        feature_selector = "f_regression"
        select_des_num = 50
        if "feature_selector" in params:
            feature_selector = params["feature_selector"]
            del params["feature_selector"]
        if "select_des_num" in params:
            select_des_num = int(params["select_des_num"])
            del params["select_des_num"]
        if (model.feature_selector_name != feature_selector) or (model.feature_select_num != select_des_num):
            model.SelectFeature(feature_selector=feature_selector, select_des_num=select_des_num)
        else:
            pass
        model.Train(estimator,params=params)
        val_metric = model.all_metrics_df.loc["mean",search_metric]
        if greater_is_better:
            return -val_metric
        else:
            return val_metric

    lr_model = LinearRegression()
    lr_params = {}

    xgbr_model = XGBRegressor(objective='reg:squarederror', random_state=random_state)
    xgbr_params = {
        'gamma': hyperopt.hp.uniform("gamma", 0, 0.5),
        'max_depth': hyperopt.hp.uniformint('max_depth', 2, 15),
        'min_child_weight': hyperopt.hp.uniformint('min_child_weight', 1, 30),
        'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.5, 1),
        'subsample': hyperopt.hp.uniform('subsample', 0.5, 1),
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.001, 0.2),
        'n_estimators': hyperopt.hp.uniformint('n_estimators', 5, 500),
         'max_delta_step':hyperopt.hp.uniform('max_delta_step', 0.5, 1),
         'reg_alpha': hyperopt.hp.uniform('reg_alpha', 0, 0.5),
         'reg_lambda': hyperopt.hp.uniform('reg_lambda', 0.5, 1),
         'scale_pos_weight': hyperopt.hp.uniform('scale_pos_weight', 0, 0.2),
    }

    rfr_model = RandomForestRegressor(random_state=random_state)
    rfr_parms = {'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
                'max_leaf_nodes': hyperopt.hp.uniformint('max_leaf_nodes', 10, 100),
                'min_samples_split': hyperopt.hp.uniformint('min_samples_split', 2, 10),
                'min_samples_leaf': hyperopt.hp.uniformint('min_samples_leaf', 1, 10),
                }

    svr_model = SVR()
    svr_params = {'C': hyperopt.hp.uniform("C", 1e-5, 1e2),
                  'gamma': hyperopt.hp.uniform("gamma", 1e-5, 1e2),
                  'epsilon': hyperopt.hp.uniform("epsilon", 1e-5, 1),
                  }

    knnD_model = KNeighborsRegressor(algorithm="auto", weights='distance')#KNUni, KNDist
    knnD_params  = {'n_neighbors': hyperopt.hp.uniformint('n_neighbors', 5, 20),
                'leaf_size': hyperopt.hp.uniformint('leaf_size', 1, 20),
                }
    knnU_model = KNeighborsRegressor(algorithm="auto", weights='uniform')#KNUni, KNDist
    knnU_params  = {'n_neighbors': hyperopt.hp.uniformint('n_neighbors', 5, 20),
                'leaf_size': hyperopt.hp.uniformint('leaf_size', 1, 20),
                }
    knnD_designed_model = KNeighborsRegressor(weights=weights_function_KNND)#algorithm="brute",  KNUni, KNDist
    knnD_designed_params  = {
        'n_neighbors': hyperopt.hp.uniformint('n_neighbors', 5, 20),
                'leaf_size': hyperopt.hp.uniformint('leaf_size', 1, 20),
                }
    lgbm_model = LGBMRegressor()
    lgbm_params  = {'num_leaves': hyperopt.hp.uniformint('num_leaves', 2, 7),
            'learning_rate': hyperopt.hp.uniform('learning_rate', 0.00001, 0.2),
            'min_child_samples': hyperopt.hp.uniformint('min_child_samples', 0, 50),
            'max_depth': hyperopt.hp.uniformint('max_depth', 0, 13),
            'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
            "bagging_fraction" :hyperopt.hp.uniform('bagging_fraction', 0.5, 1),
            }
    krr_model = KernelRidge()
    krr_params  = {'alpha': hyperopt.hp.uniform('alpha', 0, 3),
            }

    ada_model = AdaBoostRegressor() 
    ada_params  = {'learning_rate': hyperopt.hp.uniform('learning_rate', 0.00001, 0.2),
        'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
        }


    mlp_model = MLPRegressor(hidden_layer_sizes=(32,16,8))  
    mlp_params  = {
        'learning_rate_init': hyperopt.hp.uniform('learning_rate_init', 0.0001, 0.2),
        'momentum': hyperopt.hp.uniform('momentum',0.8,0.95)
        }

    model_param_dict = {"MLP": {"estimator": mlp_model, "params": mlp_params},
                        "LR": {"estimator": lr_model, "params": lr_params},
                        "XGB": {"estimator": xgbr_model, "params": xgbr_params},
                        "RF":{"estimator": rfr_model, "params": rfr_parms},
                        "SVM":{"estimator": svr_model, "params": svr_params},
                        "KNNU":{"estimator": knnU_model, "params": knnU_params},
                        "KNND":{"estimator": knnD_model, "params": knnD_params},
                        "LGBM":{"estimator": lgbm_model, "params": lgbm_params},
                        "KRR":{"estimator": krr_model, "params": krr_params},
                        "ADA":{"estimator": ada_model, "params": ada_params},
                        "KNNDde":{"estimator": knnD_designed_model, "params": knnD_designed_params},

                       }

    for m in model_list:
        estimator = model_param_dict[m]["estimator"]
        model_name = str(estimator)
        model_name = model_name[:model_name.find("(")]
        params_space = {"feature_selector": hyperopt.hp.choice('feature_selector',feature_selector_list),
                        "select_des_num":hyperopt.hp.choice("select_des_num",select_des_num_list)}
        params_space.update(model_param_dict[m]["params"])
        best_params = hyperopt.fmin(Search, space=params_space, algo=hyperopt.tpe.suggest,
                                    max_evals=search_max_evals,rstate=np.random.default_rng(random_state))
        for key,value in params_space.items():
            if value.name == "int":
                best_params[key] = int(best_params[key])
        print("Best params: ",best_params)
        select_des_num = select_des_num_list[best_params["select_des_num"]]
        feature_selector = feature_selector_list[best_params["feature_selector"]]
        model.SelectFeature(feature_selector=feature_selector, select_des_num=select_des_num, saved_dir=saved_dir, saved_file_note=model_name)
        del best_params["select_des_num"]
        del best_params["feature_selector"]
        file = open('./_/best_params.txt', 'w') 
        file.write(str(model_name))
        file.write("\n")

        for k,v in best_params.items():
            file.write(str(k)+'='+str(v)+',')
        file.close()

        model.Train(estimator,params=best_params,saved_dir=saved_dir)
        model.all_metrics_df["model_name"] = model_name
        model.all_metrics_df["feature_num"] = select_des_num
        model.all_metrics_df["random_state"] = random_state
        metrics_out_file = os.path.join(saved_dir,"model_metrics.csv")
        model.all_metrics_df.to_csv(metrics_out_file,mode="a")
        model.SaveTotalModel(saved_dir=saved_dir,saved_file_note=random_state)
        model.DrawScatter(model.val_y_all, model.val_pred_all,  saved_dir=saved_dir, saved_file_note=model_name, data_group="validation")
        if model.test_y is not None:
            model.DrawScatter(model.test_y, model.test_pred_mean, saved_dir=saved_dir, saved_file_note=model_name)

def weights_function_KNND(x):
    return 1/(x+0.31)
def create_folder(directory):
    if not os.path.exists(directory):  
        os.mkdir(directory)
        print("successfully make the directory")
    else:  
        if not os.listdir(directory):  
            print("directory exists already, no more directory is made")
            pass
        else:  
            print("something already in the directory, please rename the directory'_'")
            sys.exit()


if __name__ == "__main__":

###########Regression################

    basic_num_of_features_including_refra_list = [32,64,128,256,512,]# #require modi: 51, 101, 151, 301
    target_list = ["Calc_Kr"]
    split_method_list = ["spxy",]
    type_of_NN_list = ["GIN","GCN"] 
    #########################################
    for i in range(len(type_of_NN_list)):
        for j in basic_num_of_features_including_refra_list:
            target_file_base = "emb_{}_{}.csv".format(type_of_NN_list[i],str(j))  

            for n in range(len(target_list)):
                for l in range(len(split_method_list)):

                    t0 = time.time()
                    data_dir = "./"
                    in_file_name = ""
                    in_file_path = os.path.join(data_dir, in_file_name)
                    
                    des_type = ('')
                    create_folder("./_")

                    random_state = 42


                    feature_selector_list = ("RFE",)
                    select_des_num_list = (1000,)# a large value means employing all the features in the feature matrix

                    model_list = ("ADA","MLP","XGB","SVM","LGBM","RF","KNND","KNNDde","KRR", )
                    
                    
                    smi_column = "cn_smiles"
                    name_column = "init_id"
                    label_column = "pValue"
                    group_column = "cluster_label"
                    
                    test_size = 0.2
                    kfold_type = "normal"

                    search_max_evals = 10
                    search_metric = "val_RMSE"

                    k = 10

                    saved_dir = os.path.join(data_dir, "{}_{}".format(in_file_name[-11:],"_".join(des_type)))
                    print(saved_dir)


                    MainRegression(in_file_path, saved_dir, feature_selector_list, select_des_num_list, model_list=model_list, search_max_evals=search_max_evals, 
                                    des_type=des_type, smi_column=smi_column, name_column=name_column, label_column=label_column, group_column=group_column, k=k,
                                    test_size=test_size, kfold_type=kfold_type, random_state=random_state, search_metric=search_metric, greater_is_better=False, target_file=target_file_base,basic_num_of_features_including_refra=j,target=target_list[n],split_method=split_method_list[l])


                    
                    new_folder_name = str("Calc_kr"+ type_of_NN_list[i]) + str("_") + str(j) + str("_") + str(target_list[n])  +  str("_") + str(split_method_list[l])
                    print(cal_perf())
                    os.rename("./_", new_folder_name)
                        
                    print("Time cost: {}".format(Sec2Time(time.time()-t0)))
