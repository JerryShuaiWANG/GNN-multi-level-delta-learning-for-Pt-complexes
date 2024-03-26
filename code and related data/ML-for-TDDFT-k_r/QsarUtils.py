#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : QsarUtils.py
@Description  : 
@Time         : 2020/05/27 09:51:25
@Author       : Shengde Zhang
@Version      : 1.0
'''

import os,time,copy
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import joblib
# from rdkit import Chem
# from rdkit.Chem import AllChem,MACCSkeys
# from rdkit.Chem.SaltRemover import SaltRemover

import warnings
import matplotlib.pyplot as plt
from matplotlib import colors
plt.switch_backend('agg')
from sklearn.feature_selection import SelectKBest, f_regression,mutual_info_regression,RFE,VarianceThreshold
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,LeaveOneOut,LeaveOneGroupOut
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree


def Sec2Time(seconds):  # convert seconds to time
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return ("{:02d}h:{:02d}m:{:02d}s".format(h, m, s))

def SplitTrainTest(df,test_size=0.1, label_column="labels", group_column="", random_state=0, min_test_num=1):
    """mode: random/stratified"""
    np.random.seed(random_state)

    if group_column not in ["", None, "None", "null"]:
        if group_column not in df.columns:
            raise KeyError
        groups = df[group_column].unique()
        train_index = []
        test_index = []
        for g in groups:
            df_sub = df[df[group_column]==g]
            test_num = int(len(df_sub) * test_size)
            if test_num < min_test_num:
                train_index.extend(df_sub.index)
            else:
                min_max_idx = [df_sub[label_column].idxmin(), df_sub[label_column].idxmax()]
#                 print(df_sub.shape,min_max_idx)
                train_index.extend(min_max_idx)
                df_without_max_min = df_sub.drop(index=min_max_idx)
                train_df, test_df = train_test_split(df_without_max_min, test_size=test_size, random_state=random_state)
                test_index.extend(test_df.index)
                train_index.extend(train_df.index)
        train_index = np.random.permutation(train_index)
        test_index = np.random.permutation(test_index)
    else:
        min_max_idx = [df[label_column].idxmax(), df[label_column].idxmin()]
        max_min_df = df.iloc[min_max_idx]
        df_without_max_min = df.drop(index=min_max_idx)
        train_df, test_df = train_test_split(df_without_max_min, test_size=test_size, random_state=random_state)
        train_df = train_df.append(max_min_df)
        train_df = train_df.reindex(index=np.random.permutation(train_df.index))
        test_df = test_df.reindex(index=np.random.permutation(test_df.index))
        train_index = train_df.index.tolist()
        test_index = test_df.index.tolist()

    print('Train test split successfully: train/test = {}/{}'.format(len(train_index),len(test_index)))
    return train_index,test_index

def CalMolDescriptors(in_file_path, saved_dir=None, smi_column="Smiles", name_column="", label_column="",
                      des_type='all', sep=',', out_file_prefix=None, tmp_file_name="tmp",use_cache=False):
    """des_type: rdkit/ecfp4/macss/rdkfp/ttfp"""
    ### 输入文件处理
    # print(smi_column)
    t0 = time.time()
    init_df = pd.read_csv(in_file_path, sep=sep)
    # print(init_df.head())
    in_df = pd.DataFrame(init_df.loc[:, smi_column].values, columns=['Smiles'])

    if name_column not in ["",None,"None"]:
        in_df['Name'] = init_df.loc[:, name_column].values
    else:
        in_df['Name'] = in_df.index.astype(str) + '_' + in_df['Smiles']
        init_df['Name'] = in_df.index.astype(str) + '_' + in_df['Smiles']
        name_column = 'Name'

    parent_dir = os.path.dirname(in_file_path)
    if (saved_dir == None) or (saved_dir == "None"):
        saved_dir = parent_dir
    else:
        try:
            os.mkdir(saved_dir)
        except:
            pass

    if type(out_file_prefix) == str:
        pass
    else:
        in_file_base_name = os.path.basename(in_file_path)
        out_file_prefix = in_file_base_name[:in_file_base_name.rfind('.')]

    if type(des_type) == str:
        des_type = [des_type]
    if (type(des_type) == list) or (type(des_type) == tuple):
        for t in des_type:
            if t in [ 'rdkit', 'ecfp4', 'maccs','rdkfp','ttfp']:
                pass
            elif t == "all":
                des_type = ['rdkit', 'ecfp4', 'maccs','rdkfp','ttfp']
                break
            else:
                print("The des_type '{}' is invalid".format(t))
                des_type.remove(t)

    else:
        print("The des_type '{}' is invalid".format(des_type))
        exit(0)
    
    des_file = os.path.join(saved_dir, out_file_prefix+"_{}.csv".format("_".join(des_type)))
    if os.path.exists(des_file) and use_cache:
        df_mol_des = pd.read_csv(des_file)
        print("Find existed des file: ".format(des_file))
        print('Molecule descriptors shape: {}'.format(df_mol_des.shape))
        return df_mol_des

    invalid_id = []
    mols = []
    temp_smi_path = os.path.join(saved_dir,'{}.smi'.format(tmp_file_name))
    wt = Chem.SmilesWriter(temp_smi_path, delimiter='\t', includeHeader=False)
    for i, smi in enumerate(in_df['Smiles']):
        # print(in_df.head())
        m = Chem.MolFromSmiles(smi)
        if m:
            remover = SaltRemover()
            m = remover.StripMol(m, dontRemoveEverything=True)
            m.SetProp('_Name', str(in_df.loc[i, 'Name']))
            wt.write(m)
            mols.append(m)
        else:
            invalid_id.append(i)
    wt.close()
    print('\nNumber of input molecules: {}\n'.format(len(in_df)))
    print('Number of invalid molecules: {}\n'.format(len(invalid_id)))
    in_df.drop(labels=invalid_id,axis=0,inplace=True)
    in_df.reset_index(drop=True,inplace=True)

    # mols = Chem.SmilesMolSupplier(temp_smi_path, smilesColumn=0, nameColumn=1, delimiter='\t', titleLine=False)
    t1 = time.time()
    print('Preprocessing done, time cost: {}'.format(Sec2Time(t1 - t0)))

    # 计算描述符
    temp_csv_path = temp_smi_path[:-4] + '.csv'
    des_dfs = []
    t2 = time.time()
    if "rdkit" in des_type:
        if os.path.exists(temp_csv_path):
            try:
                os.remove(temp_csv_path)
            except Exception as e:
                print(e)
                # shutil.rmtree(temp_csv_path)
        in_params = 'smilesColumn,1,smilesNameColumn,2,smilesDelimiter,tab,smilesTitleLine,False'
        options = {'--autocorr2DExclude': 'yes',
                   '--descriptorNames': '',
                   '--examples': False,
                   '--fragmentCount': 'yes',
                   '--help': False,
                   '--infile': temp_smi_path,
                   '--infileParams': in_params,
                   '--list': False,
                   '--mode': '2D',
                   '--mp': 'no',
                   '--mpParams': 'auto',
                   '--outfile': temp_csv_path,
                   '--outfileParams': 'auto',
                   '--overwrite': False,
                   '--precision': '3',
                   '--smilesOut': 'no',
                   '--workingdir': None}
        rdcd.main(options)
        rdkit2d_df = pd.read_csv(temp_csv_path).drop(labels='Ipc', axis=1)
        des_dfs.append(rdkit2d_df)
        # rdkit2d_df.insert(1,'Smiles',in_df['Smiles'].values)
        # rdkit2d_path = os.path.join(saved_dir, '{}_rdkit.csv'.format(out_file_prefix))
        # rdkit2d_df.to_csv(rdkit2d_path, index=False)
        # print("Calculated \033[36;1mrdkit2D descriptors\033[0m have been saved in '\033[45;1m{}\033[0m'".format(
        #     rdkit2d_path))
        print("Getting \033[36;1mrdkit2D descriptors\033[0m successfully")
#         print('Time cost: {}'.format(Sec2Time(time.time() - t2)))

    if 'ecfp4' in des_type:
        t2 = time.time()
        mgfps2 = np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048,
                                                                      useChirality=True)) for mol in mols])

        mg_df2 = pd.DataFrame(
            mgfps2, columns=['ECFP4_{}'.format(i) for i in range(mgfps2.shape[1])])
        mg_df2 = pd.concat([in_df['Name'], mg_df2], axis=1, sort=False)
        # mg_df2.to_csv(os.path.join(saved_dir, out_file_prefix + "_ecfp4.csv"), index=False)
        des_dfs.append(mg_df2)

        print("Getting \033[36;1mECFP4 fingerprints\033[0m successfully")
#         print('Time cost: {}'.format(Sec2Time(time.time() - t2)))

    if 'rdkfp' in des_type:
        t2 = time.time()
        rdkfps = np.array([list(Chem.RDKFingerprint(mol, fpSize=2048)) for mol in mols])

        rdk_df = pd.DataFrame(
            rdkfps, columns=['RDK_{}'.format(i) for i in range(rdkfps.shape[1])])
        rdk_df = pd.concat([in_df['Name'], rdk_df], axis=1, sort=False)
        # mg_df2.to_csv(os.path.join(saved_dir, out_file_prefix + "_rdkfp.csv"), index=False)
        des_dfs.append(rdk_df)
        print("Getting \033[36;1mRDKfingerprints\033[0m successfully")
        # print('Time cost: {}'.format(Sec2Time(time.time() - t2)))

    if 'ttfp' in des_type:
        t2 = time.time()
        ttfps = np.array([list(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)) for mol in mols])

        tt_df = pd.DataFrame(
            ttfps, columns=['TT_{}'.format(i) for i in range(ttfps.shape[1])])
        tt_df = pd.concat([in_df['Name'], tt_df], axis=1, sort=False)
        # mg_df2.to_csv(os.path.join(saved_dir, out_file_prefix + "_rdkfp.csv"), index=False)
        des_dfs.append(tt_df)

        print("Getting \033[36;1mTopologicalTorsionFingerprints\033[0m successfully")
        # print('Time cost: {}'.format(Sec2Time(time.time() - t2)))

    if 'maccs' in des_type:
        t2 = time.time()
        maccsfp = np.array([list(MACCSkeys.GenMACCSKeys(mol)) for mol in mols])
        maccs_df = pd.DataFrame(
            maccsfp, columns=['MACCS_{}'.format(i) for i in range(maccsfp.shape[1])])
        maccs_df = pd.concat([in_df['Name'], maccs_df], axis=1, sort=False)
        # maccs_df.to_csv(os.path.join(saved_dir, out_file_prefix + "_maccs.csv"), index=False)
        des_dfs.append(maccs_df)

        print("Getting \033[36;1mMACCS fingerprints\033[0m successfully")
        # print('Time cost: {}'.format(Sec2Time(time.time() - t2)))


    if os.path.exists(temp_csv_path):
        try:
            os.remove(temp_csv_path)
        except Exception as e:
            print(e)
            # shutil.rmtree(temp_csv_path)
    if os.path.exists(temp_smi_path):
        try:
            os.remove(temp_smi_path)
        except Exception as e:
            print(e)
            # shutil.rmtree(temp_smi_path)

    df_mol_des = pd.concat(des_dfs, axis=1, sort=False).reset_index(drop=True).copy()
    cpd_idx = df_mol_des.iloc[:, 0]
    df_mol_des.drop(labels=['Name'], axis=1, inplace=True)
    df_mol_des.insert(0, 'Name', cpd_idx)
    df_mol_des.insert(1, 'Smiles', init_df.loc[:,smi_column])
    df_mol_des.dropna(axis=0, inplace=True)
    print('Molecule descriptors shape: {}'.format(df_mol_des.shape))

    if label_column not in ["", None, "None"]:
#         df_mol_des = pd.concat([df_mol_des,init_df.iloc[:,label_column-1]],axis=1)
        # assert len(df_mol_des) == len(init_df)
        label_df = init_df.loc[:,[name_column, label_column]].rename(columns={name_column: "Name", label_column: "labels"})
        des_columns = df_mol_des.columns.to_list()[2:]
        df_mol_des = df_mol_des.merge(label_df, on="Name", how="left")
        df_mol_des = df_mol_des.reindex(columns=["Name","Smiles","labels"]+des_columns)
        # df_mol_des.insert(2, 'labels', label_df)
#         df_mol_des = df_mol_des.rename(columns={init_df.columns[label_column-1]:"labels"})
    df_mol_des.to_csv(des_file,index=False)
    print("Des file saved: {}".format(des_file))

    return df_mol_des

class RegressionModel(object):
    """
    Example:
        smi_column="Smiles"
        name_column="Name"
        label_column="pIC50"
        des_type = ("rdkit", "ecfp4", "maccs", "rdkfp", "ttfp")
        in_df = pd.read_csv(in_file_path)
        df_mol_des = CalMolDescriptors(in_file_path, saved_dir=saved_dir, smi_column=smi_column,
                                      name_column=name_column, label_column=label_column,
                                      des_type=des_type)
        train_index,test_index = SplitTrainTest(df_mol_des[["Name","Smiles","labels"]],
                                                test_size=0.2, group_column="",label_column="labels")
        train_df = df_mol_des.iloc[train_index]
        test_df = df_mol_des.iloc[test_index]

        train_X = train_df.iloc[:,3:].values.astype(dtype=np.float32)
        train_y = train_df.iloc[:,2].values.astype(dtype=np.float32)
        test_X = test_df.iloc[:,3:].values.astype(dtype=np.float32)
        test_y = test_df.iloc[:,2].values.astype(dtype=np.float32)

        model = RegressionModelModel(random_state=0)
        model.LoadData(train_X, train_y, test_X, test_y, train_groups=None, des_type=des_type)
        model.ScaleFeature(saved_dir="",saved_file_note="")
        model.KFoldSplit(k=5, kfold_type='normal')
        model.SelectFeature(feature_selector="f_regression", select_des_num=50, saved_dir="", saved_file_note="")
        estimator = LinearRegression()
        params = {}
        model.Train(estimator,params=params,saved_dir="")
        model.GenerateBallTree(p=1,saved_dir=saved_dir)
        model.DrawScatter(model.test_y, model.test_pred_mean, saved_dir=saved_dir, saved_file_note="")
        model.SaveTotalModel(saved_dir=saved_dir,saved_file_note="")

    """
    def __init__(self,random_state=0):
        """"""
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.train_groups = None
        self.random_state = random_state
        self.color_list = list(colors.XKCD_COLORS.values())
        self.n_jobs = int(0.8*os.cpu_count())
        self.feature_selector_name = "null"
        self.feature_select_num = 0
        self.des_type = ("all",)
        self.metrics_list = ["R2","RMSE","MAE","squared Pearson","Spearman"]
        
    def Sec2Time(self, seconds):
        """convert seconds to time"""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return ("{:02d}h:{:02d}m:{:02d}s".format(h, m, s))
        
    def CalMetrics(self, y_true, y_pred):
        
        r2 = r2_score(y_true=y_true,y_pred=y_pred)
        rmse = mean_squared_error(y_true=y_true, y_pred=y_pred) ** 0.5
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        df_y = pd.DataFrame({"y_true":y_true,"y_pred":y_pred})

        pearson_corr = (df_y.corr().iloc[0,1])**2
        spearman_corr = df_y.corr("spearman").iloc[0,1]
        return r2, rmse, mae, pearson_corr, spearman_corr
        
    def LoadData(self, train_X, train_y, test_X=None, test_y=None, train_groups=None, des_type=None):

        self.train_X = np.array(train_X)
        self.train_y = np.array(train_y)
        if (test_X is not None) and (test_y is not None):
            self.test_X = np.array(test_X)
            self.test_y = np.array(test_y)
        self.train_groups = np.array(train_groups)
        if des_type != None:
            self.des_type = des_type
        print("="*50)
        print("Train_X shape: ", self.train_X.shape)
        print("Train_y shape: ", self.train_y.shape)
        if self.test_X is not None:
            print("Test_X shape: ",self.test_X.shape)
            print("Test_y shape: ", self.test_y.shape)
        print("=" * 50)

    def ScaleFeature(self, feature_range=(0,1), saved_dir="", saved_file_note=""):

        print("Scaled feature range: ",feature_range)
        # self.scaler = MinMaxScaler(feature_range=feature_range)
        self.scaler = StandardScaler()
        self.train_X_init = self.train_X.copy()
        self.train_X = self.scaler.fit_transform(self.train_X)
        if (self.test_X is not None) and (self.test_y is not None):
            self.test_X_init = self.test_X
            self.test_X= self.scaler.transform(self.test_X)
        if saved_dir not in ["",None,"None"]:
            scaler_x_file = os.path.join(saved_dir, "scaler_{}.pkl".format(saved_file_note))
            with open(scaler_x_file, 'wb') as f:
                joblib.dump(self.scaler, f)
        
    def SelectFeature(self, saved_dir="", feature_selector='f_regression', 
                      select_des_num=100, saved_file_note=""):
        """X: features scaled
           Selector: 'RFE', 'f_regression', or 'mutual_info_regression'
           select_num: number of features to select"""
        
        t1 = time.time()
        self.variance_filter = VarianceThreshold(threshold = 0)
        self.train_X_filtered = self.variance_filter.fit_transform(self.train_X)
        print("Features shape after variance filter: ",self.train_X_filtered.shape)
        print('Executing feature selection on features by {}.'.format(feature_selector))
        if self.train_X_filtered.shape[1] <= select_des_num:
            select_des_num = self.train_X_filtered.shape[1]
        if feature_selector == 'RFE':
            base_model = RandomForestRegressor(n_estimators=20, random_state=self.random_state, n_jobs=self.n_jobs)
            self.selector = RFE(base_model, n_features_to_select=select_des_num, step=0.01)
        elif feature_selector == 'f_regression':
            self.selector = SelectKBest(score_func=f_regression, k=select_des_num)
        elif feature_selector == 'mutual_info_regression':
            self.selector = SelectKBest(score_func=mutual_info_regression, k=select_des_num)
        else:
            raise NotImplementedError("Feature selector choice: REF/f_regression/mutual_info_regression")
        self.selector.fit(self.train_X_filtered, self.train_y)
        self.train_X_selected = self.selector.transform(self.train_X_filtered)
        
        if (self.test_X is not None) and (self.test_y is not None):
            self.test_X_filtered = self.variance_filter.transform(self.test_X)
            self.test_X_selected = self.selector.transform(self.test_X_filtered)
        self.feature_selector_name = feature_selector
        self.feature_select_num = select_des_num
        if saved_dir not in ["",None,"None"]:
            variance_file = os.path.join(saved_dir,"variance_{}.pkl".format(saved_file_note))
            with open(variance_file, 'wb') as f:
                joblib.dump(self.variance_filter, f)
            selector_file = os.path.join(saved_dir,'selector_{}.pkl'.format(saved_file_note))
            with open(selector_file, 'wb') as f:
                joblib.dump(self.selector, f)
        print("Selected feature num: ", self.train_X_selected.shape[1])
        print('Time cost for selection: {}'.format(Sec2Time(time.time()-t1)))
        print('')
        
        
    def KFoldSplit(self, k=5, kfold_type="normal"):
        """kfold_type: normal(KFold) / stratified(StratifiedKFold) / group(GroupKFold) / none(all train)"""
        np.random.seed(self.random_state)
        train_val_idxs = []
        self.k = k
        if (kfold_type == None) or (kfold_type == "none") or (k == 1):
            print("Using none KFold, all training.")
            train_idx = [i for i in range(len(self.train_X))]
            val_idx = [i for i in range(len(self.train_X))]
            train_idx = np.random.permutation(train_idx)
            val_idx = np.random.permutation(val_idx)
            train_val_idxs.append([train_idx, val_idx])

        elif (kfold_type == 'group') and (k != 1):
            print("Using GroupKFold.")
            kf = GroupKFold(k)
            for train_idx, val_idx in kf.split(self.train_X, groups=self.train_groups):
                train_idx = np.random.permutation(train_idx)
                val_idx = np.random.permutation(val_idx)
                train_val_idxs.append([train_idx, val_idx])
        elif (kfold_type == 'stratified') and (k != 1):
            print("Using StratifiedKFold.")
            kf = StratifiedKFold(k,shuffle=True, random_state=self.random_state)
            for train_idx, val_idx in kf.split(self.train_X, y=self.train_groups):
                train_idx = np.random.permutation(train_idx)
                val_idx = np.random.permutation(val_idx)
                train_val_idxs.append([train_idx, val_idx])
        elif (kfold_type == 'loo'):
            print("Using LeaveOneOut.")
            kf = LeaveOneOut()
            for train_idx, val_idx in kf.split(self.train_X):
                train_idx = np.random.permutation(train_idx)
                val_idx = np.random.permutation(val_idx)
                train_val_idxs.append([train_idx, val_idx])
        elif (kfold_type == 'logo'):
            print("Using LeaveOneGroupOut.")
            kf = LeaveOneGroupOut()
            for train_idx, val_idx in kf.split(self.train_X, groups=self.train_groups):
                train_idx = np.random.permutation(train_idx)
                val_idx = np.random.permutation(val_idx)
                train_val_idxs.append([train_idx, val_idx])
        else:
            print("Using normal KFold.")
            kfold_type = 'kfold'
            kf = KFold(k,shuffle=True,random_state=self.random_state)
            for train_idx, val_idx in kf.split(self.train_X):
                train_idx = np.random.permutation(train_idx)
                val_idx = np.random.permutation(val_idx)
                train_val_idxs.append([train_idx, val_idx])
        self.kfold_type = kfold_type
        self.train_val_idxs = train_val_idxs
    
    def Train(self, estimater, params, saved_dir=""):
        
        self.model = estimater
        self.params = params
        self.model_name = str(self.model)
        self.model_name = self.model_name[:self.model_name.find("(")]
        tick = time.time()
        self.model.set_params(**params)
        i = 0
        train_metrics_all = []
        val_metrics_all = []
        test_metrics_all = []
        test_pred_all = []
        val_pred_all = []
        val_y_all = []
        k = len(self.train_val_idxs)
        self.models = []
        for train_idx, val_idx in self.train_val_idxs:
            print("="*50)
            print("Train/validation num: {}/{}".format(len(train_idx),len(val_idx)))
            i += 1
            kf_train_X = self.train_X_selected[train_idx]
            kf_train_y = self.train_y[train_idx]
            kf_val_X = self.train_X_selected[val_idx]
            kf_val_y = self.train_y[val_idx]
            self.model.fit(kf_train_X, kf_train_y)
            self.models.append(copy.copy(self.model))
            if saved_dir not in ["",None,"None"]:
                with open(os.path.join(saved_dir, '{}_{}.pkl'.format(self.model_name, i)),'wb+') as f:
                    joblib.dump(self.model, f)
                self.VisualizeChemSpace(kf_train_X,kf_val_X,saved_dir=saved_dir,method="tSNE", notes="fold{}".format(i))
            kf_train_pred = self.model.predict(kf_train_X)
            kf_val_pred = self.model.predict(kf_val_X)
            kf_train_metrics = self.CalMetrics(kf_train_y,kf_train_pred)
            train_metrics_all.append(kf_train_metrics)
            if self.kfold_type != "loo":
                kf_val_metrics = self.CalMetrics(kf_val_y,kf_val_pred)
                val_metrics_all.append(kf_val_metrics)
            else:
                kf_val_metrics = [None]*len(self.metrics_list)
                val_metrics_all.append(kf_val_metrics)
            val_pred_all.extend(kf_val_pred)
            val_y_all.extend(kf_val_y)
#             print("Train r2_score: {:.4f}".format(kf_train_metrics[0]))
#             print("Val r2_score: {:.4f}".format(kf_val_metrics[0]))
            if (hasattr(self, "test_X_selected")) and (self.test_y is not None):
                test_pred = self.model.predict(self.test_X_selected)
                # print("Training {} test_pred: {}".format(i, test_pred[0]))
                test_pred_all.append(test_pred)
                test_metrics = self.CalMetrics(y_true=self.test_y, y_pred=test_pred)
                test_metrics_all.append(test_metrics)
#                 print('Test r2_score: {:.4f}'.format(test_metrics[0]))
            
            # tock = time.time()
#             print('Running time for fold {}: {}'.format(i, self.Sec2Time(tock - tick)))
#             print("=" * 50)
        
        metrics_list = self.metrics_list
        self.train_metrics_df = pd.DataFrame(train_metrics_all,columns=["tr_"+s for s in metrics_list],
                                        index=['fold_{}'.format(i+1) for i in range(k)])
        self.val_metrics_df = pd.DataFrame(val_metrics_all,columns=["val_"+s for s in metrics_list],
                                      index=['fold_{}'.format(i+1) for i in range(k)])
        metrics_dfs = [self.train_metrics_df, self.val_metrics_df]

        self.val_pred_all = np.array(val_pred_all, dtype=np.float32)
        self.val_y_all = np.array(val_y_all, dtype=np.float32)
        np.savetxt('./_/val_pred_all.csv', self.val_pred_all, delimiter=',')
        np.savetxt('./_/val_y_all.csv', self.val_y_all, delimiter=',')
        
        if (hasattr(self, "test_X_selected")) and (self.test_y is not None):
                        
            self.test_metrics_df = pd.DataFrame(test_metrics_all,columns=["te_"+s for s in metrics_list],
                                index=['fold_{}'.format(i+1) for i in range(k)])
            metrics_dfs.append(self.test_metrics_df)
            self.test_pred_mean = np.mean(test_pred_all, axis=0)
            df_te_pred = pd.DataFrame([self.test_y.squeeze(), self.test_pred_mean.squeeze()],
                          index=['true_value',"pred_value"]).T
            if saved_dir not in ["",None,"None"]:
                test_pred_file = os.path.join(saved_dir, "test_predicted_{}.csv".format(self.model_name))
#                 print(test_pred_file)
                df_te_pred.to_csv(test_pred_file,index=False)
        all_metrics_df = pd.concat(metrics_dfs,axis=1).T
        all_metrics_df['mean'] = all_metrics_df.mean(axis=1)
        all_metrics_df['std'] = all_metrics_df.std(axis=1)
        
        if self.kfold_type == "loo":
            val_mean_metric = self.CalMetrics(self.val_y_all, self.val_pred_all)
            print(val_mean_metric)
            for c_idx, col in enumerate(["val_"+s for s in metrics_list]):
                all_metrics_df.loc[col, 'mean'] = val_mean_metric[c_idx]
        self.all_metrics_df = all_metrics_df.T
        print('*' * 50)
        print("All results for k-fold cross validation: ")
        print(self.all_metrics_df[[col for col in self.all_metrics_df.columns if (("R2" in col) or ("RMSE" in col))]])
        print('Total run time: {}'.format(self.Sec2Time(time.time()-tick)))

    def SaveTotalModel(self,saved_dir,saved_file_note=""):
        total_model_file = os.path.join(saved_dir,"total_model_{}{}_{}{}_{}_{}.model".format(self.feature_selector_name,
                                                                                             self.feature_select_num,
                                                                                             self.kfold_type,
                                                                                             self.k,
                                                                                             self.model_name,
                                                                                             saved_file_note))
        with open(total_model_file, "wb+") as f:
            joblib.dump(self, f)
        print("The total model file has been saved: {}".format(total_model_file))

    def LoadTotalModel(self,model_file):
        with open(model_file,'rb+') as f:
            new_model = joblib.load(f)
        for key,value in new_model.__dict__.items():
            self.__setattr__(key,value)

    def DrawScatter(self, y_true, y_pred, saved_dir, saved_file_note="", data_group="test"):
        xy_min = min(y_true.min(),y_pred.min())
        xy_max = max(y_true.max(),y_pred.max())
        fig_file = os.path.join(saved_dir, "{}_predicted_{}.png".format(data_group, saved_file_note))
        plt.clf()
        plt.figure(figsize=(6,6))
        plt.plot(y_true,y_pred,marker='x', linestyle='',color=self.color_list[-1])
        plt.plot([xy_min-0.1,xy_max+0.1],[xy_min-0.1,xy_max+0.1],color=self.color_list[-2])
        plt.xlim([xy_min-0.1,xy_max+0.1])
        plt.ylim([xy_min-0.1,xy_max+0.1])
        plt.xlabel('y_true', fontdict={'fontsize': 15})
        plt.ylabel('y_pred', fontdict={'fontsize': 15})
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def GenerateBallTree(self, p=1, neighbors_num=1, saved_dir=""):

        self.balltrees = []
        self.ref_dist_values = []
        self.feature_weights_list = []
        for i, (train_idx, val_idx) in enumerate(self.train_val_idxs):
            model = self.models[i]
            if hasattr(model, "coef_"):
                feature_weights = model.coef_
            elif hasattr(model, "feature_importances_"):
                feature_weights = model.feature_importances_
            else:
                warnings.warn("The trained models have no attribute like 'coef_' or 'feature_importances_'")
                feature_weights = np.array([1]*self.train_X_selected.shape[1])
            scaler_w = MinMaxScaler(feature_range=(0.1,0.9))
            feature_weights = scaler_w.fit_transform(feature_weights.reshape(-1,1)).squeeze()
            self.feature_weights_list.append(feature_weights)
            kf_train_X = self.train_X_selected[train_idx]
            kf_val_X = self.train_X_selected[val_idx]
            balltree = BallTree(kf_train_X, metric='wminkowski', p=p, w=feature_weights)
            if saved_dir not in ["",None,"None"]:
                with open(os.path.join(saved_dir, "balltree_p{}_fold_{}.pkl".format(p, i+1)), "wb+") as f:
                    joblib.dump(balltree, f)
            dist, ind = balltree.query(kf_val_X, k=neighbors_num, dualtree=True)
            dist_mean = dist.mean(axis=1)
            ref_dist_value = 2.5*np.quantile(dist_mean, 0.75) - 1.5*np.quantile(dist_mean, 0.25)  ### Q3+1.5IQR
            self.ref_dist_values.append(ref_dist_value)
            print("*"*50)
            print("Get reference distance value from validation set of fold {}: \033[36;1m{}\033[0m".format(i+1,ref_dist_value))
            self.balltrees.append(balltree)

    def VisualizeChemSpace(self, train_X, test_X, saved_dir, method="tSNE", notes=""):
        """method: tSNE/PCA"""
        all_X = np.concatenate((train_X,test_X),axis=0)

        if method == "tSNE":
            dimension_model = TSNE(n_components=2, perplexity=30, random_state=self.random_state)
            dimension_model.fit(all_X)
            reduction_X = dimension_model.embedding_
        elif method == "PCA":
            dimension_model = PCA(n_components=2)
            dimension_model.fit(all_X)
            reduction_X = dimension_model.transform(all_X)
        else:
            raise NotImplementedError

        dimension_model_name = str(dimension_model)
        dimension_model_name = dimension_model_name[:dimension_model_name.find("(")]
        reduction_X_tr = reduction_X[:len(train_X)]
        reduction_X_te = reduction_X[len(train_X):]
        plt.clf()
        plt.figure(figsize=(6,6))
        plt.plot(reduction_X_tr[:,0], reduction_X_tr[:,1],linestyle='',marker='+',
                 color=self.color_list[-1],markerfacecolor='w',markersize=8,label="training set")
        plt.plot(reduction_X_te[:,0], reduction_X_te[:,1],linestyle='',marker='o',
                 color=self.color_list[-2], markerfacecolor='w',markersize=6,label="test set")
        plt.xlabel("{}1".format(dimension_model_name),fontdict={'fontsize': 15})
        plt.ylabel("{}2".format(dimension_model_name),fontdict={'fontsize': 15})
        plt.legend(loc="best")
        plt.savefig(os.path.join(saved_dir,"train_test_distribution_{}_{}.png".format(dimension_model_name, notes)), 
                    dpi=300, bbox_inches="tight")
        plt.close()

    def Predit(self, X, cal_feature_distance=False, neighbors_num=1):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if hasattr(self,"scaler"):
            X = self.scaler.transform(X)
        if hasattr(self,"variance_filter"):
            X = self.variance_filter.transform(X)
        if hasattr(self,"selector"):
            X = self.selector.transform(X)
        all_y_pred = []
        for model in self.models:
            y_pred = model.predict(X)
            # print("Predicting test_pred: {}".format(y_pred[0]))
            all_y_pred.append(y_pred)
        y_pred_mean = np.mean(all_y_pred, axis=0)
        y_pred_mean = y_pred_mean.reshape((-1,1))
        if cal_feature_distance and hasattr(self,"balltrees"):
            dist_means = []
            confident_indexs = []
            for i, balltree in enumerate(self.balltrees):
                dist, ind = balltree.query(X, k=neighbors_num, dualtree=True)
                dist_mean = dist.mean(axis=1).reshape((-1,1))
                dist_means.append(dist_mean)
                confident_index = self.ref_dist_values[i] / (dist_mean + 0.00001)
                confident_indexs.append(confident_index)
            dist_mean = np.mean(dist_means, axis=0)
            confident_index = np.mean(confident_indexs, axis=0)
            y_pred_mean = np.hstack([y_pred_mean, dist_mean, confident_index])
        return y_pred_mean
    def Predit_all(self, X, cal_feature_distance=False, neighbors_num=1):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if hasattr(self,"scaler"):
            X = self.scaler.transform(X)
        if hasattr(self,"variance_filter"):
            X = self.variance_filter.transform(X)
        if hasattr(self,"selector"):
            X = self.selector.transform(X)
        all_y_pred = []
        for model in self.models:
            y_pred = model.predict(X)
            # print("Predicting test_pred: {}".format(y_pred[0]))
            all_y_pred.append(y_pred)
        y_pred_mean = np.mean(all_y_pred, axis=0)
        y_pred_mean = y_pred_mean.reshape((-1,1))
        if cal_feature_distance and hasattr(self,"balltrees"):
            dist_means = []
            confident_indexs = []
            for i, balltree in enumerate(self.balltrees):
                dist, ind = balltree.query(X, k=neighbors_num, dualtree=True)
                dist_mean = dist.mean(axis=1).reshape((-1,1))
                dist_means.append(dist_mean)
                confident_index = self.ref_dist_values[i] / (dist_mean + 0.00001)
                confident_indexs.append(confident_index)
            dist_mean = np.mean(dist_means, axis=0)
            confident_index = np.mean(confident_indexs, axis=0)
            y_pred_mean = np.hstack([y_pred_mean, dist_mean, confident_index])
        return pd.DataFrame(all_y_pred).T


    def CalDesAndPredict(self, smiles, names=None, des_type=None, cal_feature_distance=False, neighbors_num=1, tmp_file_name="temp_in"):
        """
        :param smiles: smiles string or list
        :param names: name string or list
        :param des_type: tuple, One or several in (rdkit, ecfp4, macss, rdkfp, ttfp, all)
        :param cal_feature_distance: bool
        :param neighbors_num: int
        :return: dataframe of predict results
        """
        if type(smiles) == str:
            smiles = [smiles]
        if names not in [None, "None", ""]:
            if type(names) == str:
                names = [names]
        else:
            names = ["mol_{}".format(i+1) for i in range(len(smiles))]
        df_in = pd.DataFrame([smiles,names],index=["Smiles","Name"]).T
        saved_dir = os.getcwd()
        tmp_in_file = os.path.join(saved_dir,"{}.csv".format(tmp_file_name))
        df_in.to_csv(tmp_in_file,index=False)
        if des_type != None:
            self.des_type = des_type
        df_mol_des = CalMolDescriptors(tmp_in_file, smi_column="Smiles", name_column="Name", des_type=self.des_type, use_cache=False)
        y_pred_mean = self.Predit(df_mol_des.iloc[:,2:].values, cal_feature_distance=cal_feature_distance, neighbors_num=neighbors_num)

        assert len(y_pred_mean == len(df_mol_des))
        out_df = df_mol_des.iloc[:,:2].copy()
        resut_columns = ["pred_value", "dist_value", "confident_index"]
        for c in range(y_pred_mean.shape[1]):
            out_df[resut_columns[c]] = y_pred_mean[:,c]

        if os.path.exists(tmp_in_file):
            try:
                os.remove(tmp_in_file)
            except Exception as e:
                print(e)

        return out_df

class ClassificationModel(RegressionModel):
    """
    Example:
        smi_column="Smiles"
        name_column="Name"
        label_column="Class"
        des_type = ("rdkit", "ecfp4", "maccs", "rdkfp", "ttfp")
        in_df = pd.read_csv(in_file_path)
        df_mol_des = CalMolDescriptors(in_file_path, saved_dir=saved_dir, smi_column=smi_column,
                                      name_column=name_column, label_column=label_column,
                                      des_type=des_type)
        train_index,test_index = SplitTrainTest(df_mol_des[["Name","Smiles","labels"]],
                                                test_size=0.2, group_column="",label_column="labels")
        train_df = df_mol_des.iloc[train_index]
        test_df = df_mol_des.iloc[test_index]

        train_X = train_df.iloc[:,3:].values.astype(dtype=np.float32)
        train_y = train_df.iloc[:,2].values.astype(dtype=np.float32)
        test_X = test_df.iloc[:,3:].values.astype(dtype=np.float32)
        test_y = test_df.iloc[:,2].values.astype(dtype=np.float32)

        model = ClassificationModel(random_state=0)
        model.LoadData(train_X, train_y, test_X, test_y, train_groups=None, des_type=des_type)
        model.ScaleFeature(saved_dir="",saved_file_note="")
        model.KFoldSplit(k=5, kfold_type='normal')
        model.SelectFeature(feature_selector="f_regression", select_des_num=50, saved_dir="", saved_file_note="")
        estimator = LogisticRegression()
        params = {}
        model.Train(estimator,params=params,saved_dir="")
        model.GenerateBallTree(p=1,saved_dir=saved_dir)
        model.SaveTotalModel(saved_dir=saved_dir,saved_file_note="")

    """
    def __init__(self,random_state=0):
        """"""
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.train_groups = None
        self.random_state = random_state
        self.color_list = list(colors.XKCD_COLORS.values())
        self.n_jobs = int(0.8*os.cpu_count())
        self.feature_selector_name = "null"
        self.feature_select_num = 0
        self.des_type = ("all",)
        self.prob_thd = 0.5
        self.metrics_list = ["ACC","MCC","AUC","SE","SP","TP","FP","TN","FN","CE"]

    def CalMetrics(self, y_true, y_pred, is_prob=False, prob_thd=0.5):

        def WeightCrossEntropy(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            n_pos = y_true.sum()
            n_neg = len(y_true) - n_pos
            w_pos = n_neg/len(y_true)
            w_neg = n_pos/len(y_true)
            # print(w_pos, w_neg)
            y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
            ce = (-w_pos*(1-y_true)*np.log(1-y_pred) - w_neg*y_true*np.log(y_pred)).sum()
            ce = ce/len(y_true)
            return ce

        y_true=np.array(y_true, dtype=np.int8)
        y_pred=np.array(y_pred, )
        # print("y_true shape: ", y_true.shape)
        # print("y_pred shape: ", y_pred.shape)
        y_pred_prob = y_pred.copy()
        if len(y_pred_prob.shape) == 2:
            y_pred = y_pred_prob.argmax(axis=1)
            y_pred_prob = y_pred_prob[:, 1]

        elif (is_prob == True) and (len(y_pred_prob.shape) == 1):
            y_pred = (y_pred_prob >= prob_thd).astype(np.int8)

        # print("y_true: ", y_true)
        # print("y_pred: ", y_pred)
        # print("y_pred_prob: ", y_pred_prob)
        accuracy = round(accuracy_score(y_true, y_pred), 4)
        mcc = round(matthews_corrcoef(y_true, y_pred), 4)
        auc = round(roc_auc_score(y_true, y_pred_prob), 4)
        ce = WeightCrossEntropy(y_true, y_pred_prob)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                tp += 1
            if y_true[i] == 1 and y_pred[i] == 0:
                fn += 1
            if y_true[i] == 0 and y_pred[i] == 1:
                fp += 1
            if y_true[i] == 0 and y_pred[i] == 0:
                tn += 1
        se = round(float(tp) / float(tp + fn),4)
        sp = round(float(tn) / float(tn + fp),4)
        se_sp_diff = np.abs(se - sp)

        return accuracy, mcc, auc, se, sp, tp, fp, tn, fn, ce+se_sp_diff

    def SelectFeature(self, saved_dir="", feature_selector='f_classif', 
                      select_des_num=100, saved_file_note=""):
                      
        """X: features scaled
           Selector: 'RFE', 'f_classif', or 'mutual_info_classif'
           select_num: number of features to select"""
        
        t1 = time.time()
        self.variance_filter = VarianceThreshold(threshold = 0)
        self.train_X_filtered = self.variance_filter.fit_transform(self.train_X)
        print("Features shape after variance filter: ",self.train_X_filtered.shape)
        print('Executing feature selection on features by {}.'.format(feature_selector))
        if self.train_X_filtered.shape[1] <= select_des_num:
            select_des_num = self.train_X_filtered.shape[1]
        if feature_selector == 'RFE':
            base_model = RandomForestClassifier(n_estimators=20, random_state=self.random_state, n_jobs=self.n_jobs)
            self.selector = RFE(base_model, n_features_to_select=select_des_num, step=0.01)
        elif feature_selector == 'f_classif':
            self.selector = SelectKBest(score_func=f_classif, k=select_des_num)
        elif feature_selector == 'mutual_info_classif':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=select_des_num)
        else:
            raise NotImplementedError("Feature selector choice: REF/f_classif/mutual_info_classif")
        self.selector.fit(self.train_X_filtered, self.train_y)
        self.train_X_selected = self.selector.transform(self.train_X_filtered)
        
        if (self.test_X is not None) and (self.test_y is not None):
            self.test_X_filtered = self.variance_filter.transform(self.test_X)
            self.test_X_selected = self.selector.transform(self.test_X_filtered)
        self.feature_selector_name = feature_selector
        self.feature_select_num = select_des_num
        if saved_dir not in ["",None,"None"]:
            variance_file = os.path.join(saved_dir,"variance_{}.pkl".format(saved_file_note))
            with open(variance_file, 'wb') as f:
                joblib.dump(self.variance_filter, f)
            selector_file = os.path.join(saved_dir,'selector_{}.pkl'.format(saved_file_note))
            with open(selector_file, 'wb') as f:
                joblib.dump(self.selector, f)
        print("Selected feature num: ", self.train_X_selected.shape[1])
        print('Time cost for selection: {}'.format(Sec2Time(time.time()-t1)))
        print('')

    def Train(self, estimater, params, saved_dir=""):
        
        self.model = estimater
        self.params = params
        self.model_name = str(self.model)
        self.model_name = self.model_name[:self.model_name.find("(")]
        tick = time.time()
        self.model.set_params(**params)
        i = 0
        train_metrics_all = []
        val_metrics_all = []
        test_metrics_all = []
        test_pred_all = []
        val_pred_all = []
        val_y_all = []
        k = len(self.train_val_idxs)
        self.models = []
        for train_idx, val_idx in self.train_val_idxs:
            print("="*50)
            print("Train/validation num: {}/{}".format(len(train_idx),len(val_idx)))
            # print(self.train_y)
            i += 1
            kf_train_X = self.train_X_selected[train_idx]
            kf_train_y = self.train_y[train_idx]
            kf_val_X = self.train_X_selected[val_idx]
            kf_val_y = self.train_y[val_idx]

            # print("KF_train_X/y shape: ", kf_train_X.shape, kf_train_y.shape)
            # print("KF_val_X/y shape: ", kf_val_X.shape, kf_val_y.shape)
            self.model.fit(kf_train_X, kf_train_y)
            self.models.append(copy.copy(self.model))
            if saved_dir not in ["",None,"None"]:
                with open(os.path.join(saved_dir, '{}_{}.pkl'.format(self.model_name, i)),'wb+') as f:
                    joblib.dump(self.model, f)
                self.VisualizeChemSpace(kf_train_X,kf_val_X,saved_dir=saved_dir,method="tSNE", notes="fold{}".format(i))
            if self.model_name == "SVC":
                kf_train_pred = self.model.decision_function(kf_train_X)
                kf_val_pred = self.model.decision_function(kf_val_X)
                kf_train_pred = np.stack((-kf_train_pred, kf_train_pred),axis=1)  ##change shape(n,) to shape(n,2)
                kf_val_pred = np.stack((-kf_val_pred, kf_val_pred),axis=1)
                self.prob_thd = 0
            else:
                kf_train_pred = self.model.predict_proba(kf_train_X)
                kf_val_pred = self.model.predict_proba(kf_val_X)
            kf_train_metrics = self.CalMetrics(kf_train_y,kf_train_pred,is_prob=True, prob_thd=self.prob_thd)
            train_metrics_all.append(kf_train_metrics)
            if self.kfold_type != "loo":
                kf_val_metrics = self.CalMetrics(kf_val_y,kf_val_pred, is_prob=True, prob_thd=self.prob_thd)
                val_metrics_all.append(kf_val_metrics)
            else:
                kf_val_metrics = [None]*len(self.metrics_list)
                val_metrics_all.append(kf_val_metrics)
            val_pred_all.extend(kf_val_pred)
            val_y_all.extend(kf_val_y)
#             print("Train r2_score: {:.4f}".format(kf_train_metrics[0]))
#             print("Val r2_score: {:.4f}".format(kf_val_metrics[0]))
            if (hasattr(self, "test_X_selected")) and (self.test_y is not None):
                if self.model_name == "SVC":
                    test_pred = self.model.decision_function(self.test_X_selected)
                    test_pred = np.stack((-test_pred, test_pred), axis=1)
                else:
                    test_pred = self.model.predict_proba(self.test_X_selected)
                # print("Training {} test_pred: {}".format(i, test_pred[0]))
                test_pred_all.append(test_pred)
                test_metrics = self.CalMetrics(y_true=self.test_y, y_pred=test_pred, is_prob=True, prob_thd=self.prob_thd)
                test_metrics_all.append(test_metrics)
#                 print('Test r2_score: {:.4f}'.format(test_metrics[0]))
            
            # tock = time.time()
#             print('Running time for fold {}: {}'.format(i, self.Sec2Time(tock - tick)))
#             print("=" * 50)
        
        metrics_list = self.metrics_list
        self.train_metrics_df = pd.DataFrame(train_metrics_all,columns=["tr_"+s for s in metrics_list],
                                        index=['fold_{}'.format(i+1) for i in range(k)])
        self.val_metrics_df = pd.DataFrame(val_metrics_all,columns=["val_"+s for s in metrics_list],
                                      index=['fold_{}'.format(i+1) for i in range(k)])
        metrics_dfs = [self.train_metrics_df, self.val_metrics_df]

        self.val_pred_all = np.array(val_pred_all, dtype=np.float32)
        self.val_y_all = np.array(val_y_all, dtype=np.float32)

        if (hasattr(self, "test_X_selected")) and (self.test_y is not None):
            self.test_metrics_df = pd.DataFrame(test_metrics_all,columns=["te_"+s for s in metrics_list],
                                index=['fold_{}'.format(i+1) for i in range(k)])
            metrics_dfs.append(self.test_metrics_df)
            self.test_pred_mean = np.mean(test_pred_all, axis=0)
            df_te_pred = pd.DataFrame([self.test_y.squeeze(), self.test_pred_mean.squeeze()],
                          index=['true_value',"pred_value"]).T
            if saved_dir not in ["",None,"None"]:
                test_pred_file = os.path.join(saved_dir, "test_predicted_{}.csv".format(self.model_name))
#                 print(test_pred_file)
                df_te_pred.to_csv(test_pred_file,index=False)
        all_metrics_df = pd.concat(metrics_dfs,axis=1).T
        all_metrics_df['mean'] = all_metrics_df.mean(axis=1)
        if self.kfold_type == "loo":
            val_mean_metric = self.CalMetrics(self.val_y_all, self.val_pred_all)
            for c_idx, col in enumerate(["val_"+s for s in metrics_list]):
                all_metrics_df.loc[col, 'mean'] = val_mean_metric[c_idx]
        self.all_metrics_df = all_metrics_df.T
        print('*' * 50)
        print("All results for k-fold cross validation: ")
        print(self.all_metrics_df[[col for col in self.all_metrics_df.columns if ("tr" not in col and col.split("_")[1] in ["MCC","AUC","SE","SP","CE"])]])
        print('Total run time: {}'.format(self.Sec2Time(time.time()-tick)))


    def Predit(self, X, return_prob=False, cal_feature_distance=False, neighbors_num=1):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape((1, -1))
        if hasattr(self,"scaler"):
            X = self.scaler.transform(X)
        if hasattr(self,"variance_filter"):
            X = self.variance_filter.transform(X)
        if hasattr(self,"selector"):
            X = self.selector.transform(X)
        all_y_pred = []
        for model in self.models:
            if self.model_name == "SVC":
                y_pred = model.decision_function(X)
                y_pred = np.stack((-y_pred, y_pred), axis=1)
            else:
                y_pred = model.predict_proba(X)
            # print("Predicting test_pred: {}".format(y_pred[0]))
            all_y_pred.append(y_pred)
        y_pred_mean = np.mean(all_y_pred, axis=0)
        if not return_prob:
            y_pred_mean = y_pred_mean.argmax(axis=1)  ## prob -> class_label
            y_pred_mean = y_pred_mean.reshape((-1,1))
        if cal_feature_distance and hasattr(self,"balltree"):
            dist, ind = self.balltree.query(X, k=neighbors_num, dualtree=True)
            dist_mean = dist.mean(axis=1).reshape((-1,1))
            y_pred_mean = np.hstack([y_pred_mean, dist_mean])
            if hasattr(self,"ref_dist_value"):
                confident_index = self.ref_dist_value / (dist_mean + 0.00001)
                y_pred_mean = np.hstack([y_pred_mean, confident_index])
        return y_pred_mean

if __name__ == "__main__":
    pass