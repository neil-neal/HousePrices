#!/usr/bin/env python

"""
Kaggle competition on House sale price prediction:
Summary: regression problem; 1460 training sample;
    67 numerical and categorical features
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/kernels
"""

import copy
import numpy as np
import pandas as pd
import sklearn as skl
import xgboost as xgb

def XGB_train(dfe):
    '''
    train with input samples in 3 steps:
        1. encode categorical data to numerical
        2. normalize the scale of input values
        3. train model with Gradient Boosting Trees using XGBoost
    :param dfe:
        training sample, with regression target being "SalePrice"
    :return:
        mdl: XGBoost model,
        cat_encoder: categorical feature encoder,
        sclr: SalePrice rescalor
    '''

def encode_categoricals(dfe, cat_encoder=None):
    '''
    encode categorical variables based on domain knowledge
    some simple feature engineering
    :param
      dfe:
        input dataframe to be encoded
      cat_encoder:
        dict of {categorical variable: encoded value}.
        generated from dfe if not provided
    :return:
        encoded dfe, cat_encoder
    '''

    if cat_encoder is None:
        cat_encoder=dict()

        itm = 'MSZoning'
        ecocde = {'RL': 0, 'FV': 1, 'RM': 2, 'RH': 3, 'C (all)': 4}
        cat_encoder[itm] = ecocde

        itm = 'Street'
        dfe[itm].fillna(0, inplace=True)
        ecocde = {'Pave': 0, 'Grvl': 1}
        cat_encoder[itm] = ecocde

        itm = 'Alley'
        ecocde = {np.nan: 0, 'Pave': 1, 'Grvl': 2}
        cat_encoder[itm] = ecocde

        itm = 'LandContour'
        ecocde = {'Low': 0, 'Lvl': 1, 'Bnk': 2, 'HLS': 3}
        cat_encoder[itm] = ecocde

        # we may want to remove this feature
        itm = 'Utilities'
        ecocde = {'AllPub': 0, 'NoSeWa': 1}
        cat_encoder[itm] = ecocde

        itm = 'LotConfig'
        ecocde = {'Inside': 0, 'CulDSac': 1, 'Corner': 2, 'FR2': 3, 'FR3': 4}
        cat_encoder[itm] = ecocde

        itm = 'LandSlope'
        ecocde = {'Gtl': 0, 'Mod': 1, 'Sev': 2}
        cat_encoder[itm] = ecocde

        itm = 'Neighborhood'
        mn = dfe['SalePrice'].median()
        encode = (dfe.groupby(itm)['SalePrice'].median() - mn).to_dict()
        encode.update({np.nan: 1})
        cat_encoder[itm] = encode

        itm = 'Condition1'
        mn = dfe['SalePrice'].median()
        encode = (dfe.groupby(itm)['SalePrice'].median() - mn).to_dict()
        encode.update({np.nan: 1})
        cat_encoder[itm] = encode

        itm = 'Condition2'
        mn = dfe['SalePrice'].median()
        encode = (dfe.groupby(itm)['SalePrice'].median() - mn).to_dict()
        encode.update({np.nan: 1})
        cat_encoder[itm] = encode

        itm = 'BldgType'
        ecocde = {'Twnhs': 0, 'TwnhsE': 1, 'Duplex': 2, '2fmCon': 3, '1Fam': 4}
        cat_encoder[itm] = ecocde

        itm = 'HouseStyle'
        mn = dfe['SalePrice'].median()
        encode = (dfe.groupby(itm)['SalePrice'].median() - mn).to_dict()
        encode.update({np.nan: 1})
        cat_encoder[itm] = encode

        itm = 'RoofStyle'
        ecocde = {'Gable': 0, 'Hip': 1, 'Flat': 3, 'Gambrel': 4, 'Mansard': 5, 'Shed': 6}
        cat_encoder[itm] = ecocde

        itm = 'Exterior1st'
        mn = dfe['SalePrice'].median()
        encode = (dfe.groupby(itm)['SalePrice'].median() - mn).to_dict()
        encode.update({np.nan: 1})
        cat_encoder[itm] = encode

        itm = 'Exterior2nd'
        mn = dfe['SalePrice'].median()
        encode = (dfe.groupby(itm)['SalePrice'].median() - mn).to_dict()
        encode.update({np.nan: 1})
        cat_encoder[itm] = encode

        itm='MasVnrType'
        ecocde = {'None': 0, np.nan: 0, 'BrkCmn':1, 'BrkFace':2, 'Stone':3}
        cat_encoder[itm] = ecocde

        itm = 'ExterQual'
        ecocde = {'Po': -2, 'Fa': -1, 'TA': 0, 'Gd': 1, 'Ex': 2}
        cat_encoder[itm] = ecocde

        itm = 'ExterCond'
        ecocde = {'Po': -2, 'Fa': -1, 'TA': 0, 'Gd': 1, 'Ex': 2}
        cat_encoder[itm] = ecocde

        itm = 'Foundation'
        ecocde = {'Wood': -2, 'BrkTil': -1, 'Slab': 0, 'Stone': 1, 'CBlock': 2, 'PConc': 3}
        cat_encoder[itm] = ecocde

        itm = 'BsmtQual'
        ecocde = {'Po': -2, 'Fa': -1, 'TA': 0, 'Gd': 1, 'Ex': 2, np.nan: 0}
        cat_encoder[itm] = ecocde

        itm = 'BsmtCond'
        ecocde = {'Po': -2, 'Fa': -1, 'TA': 0, 'Gd': 1, 'Ex': 2, np.nan: 0}
        cat_encoder[itm] = ecocde

        itm = 'BsmtExposure'
        ecocde = {'No': 1, 'Mn': 1, 'Av': 2, 'Gd': 3, np.nan: 0}
        cat_encoder[itm] = ecocde

        for itm in ['BsmtFinType1', 'BsmtFinType2']:
            ecocde = {'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5}
            cat_encoder[itm] = ecocde

        itm = 'HeatingQC'
        ecocde = {'Po': -2, 'Fa': -1, 'TA': 0, 'Gd': 1, 'Ex': 2}
        cat_encoder[itm] = ecocde

        itm = 'CentralAir'
        ecocde = {'N': 0, 'Y': 1}
        cat_encoder[itm] = ecocde

        itm = 'Electrical'
        ecocde = {'Mix': -2, 'FuseP': -1, 'FuseF': 0, 'FuseA': 1, 'SBrkr': 2, np.nan: 0}
        cat_encoder[itm] = ecocde

        itm = 'KitchenQual'
        ecocde = {'Po': -2, 'Fa': -1, 'TA': 0, 'Gd': 1, 'Ex': 2, np.nan: 0}
        cat_encoder[itm] = ecocde

        itm = 'Functional'
        ecocde = {'Typ': 0, 'Min1': -1, 'Min2': -2, 'Mod': -3, 'Maj1': -4, 'Maj2': -5, 'Sev': -6, 'Sal': -7}
        cat_encoder[itm] = ecocde

        itm = 'FireplaceQu'
        ecocde = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, np.nan: 0}
        cat_encoder[itm] = ecocde

        itm = 'GarageType'
        mn = dfe['SalePrice'].median()
        encode = (dfe.groupby(itm)['SalePrice'].median() - mn).to_dict()
        encode.update({np.nan: 1})
        cat_encoder[itm] = encode

        itm = 'GarageFinish'
        ecocde = {np.nan: 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
        cat_encoder[itm] = ecocde

        itm = 'GarageQual'
        ecocde = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, np.nan: 0}
        cat_encoder[itm] = ecocde

        itm = 'GarageCond'
        ecocde = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, np.nan: 0}
        cat_encoder[itm] = ecocde

        itm = 'PavedDrive'
        ecocde = {'N': 0, 'P': 1, 'Y': 2}
        cat_encoder[itm] = ecocde

        itm = 'PoolQC'
        ecocde = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, np.nan: 0}
        cat_encoder[itm] = ecocde

        itm = 'Fence'
        mn = dfe['SalePrice'].median()
        encode = (dfe.groupby(itm)['SalePrice'].median() - mn).to_dict()
        encode.update({np.nan: 0})
        cat_encoder[itm] = encode

        itm = 'MiscFeature'
        mn = dfe['SalePrice'].median()
        encode = (dfe.groupby(itm)['SalePrice'].median() - mn).to_dict()
        encode.update({np.nan: 0})
        cat_encoder[itm] = encode

        itm = 'SaleType'
        mn = dfe['SalePrice'].median()  # better domain knowledge needed
        encode = (dfe.groupby(itm)['SalePrice'].median() - mn).to_dict()
        encode.update({np.nan: 0})
        cat_encoder[itm] = encode

        itm = 'SaleCondition'
        mn = dfe['SalePrice'].median()
        encode = (dfe.groupby(itm)['SalePrice'].median() - mn).to_dict()
        encode.update({np.nan: 0})
        cat_encoder[itm] = encode

    for itm, ecd in cat_encoder.items():
        dfe[itm].replace(ecd, inplace=True)

    # some categorical values did not show up in training set.
    dfe.replace({r'\.*': 0, '\n': 0, 'AsphShn': 0, 'RRNe': 0, 'PosA': 0, 'RRAn': 0}, inplace=True, regex=True)

    dfe['Condition'] = dfe['Condition1'] + dfe['Condition2']
    dfe['Exterior'] = dfe['Exterior1st'] + dfe['Exterior2nd']
    dfe['BsmtFin'] = (dfe['BsmtFinType1'] * dfe['BsmtFinSF1'] + dfe['BsmtFinType2'] * dfe['BsmtFinSF2']) / dfe[
        'TotalBsmtSF']

    dfe.drop(['Heating', 'Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd', 'BsmtFinType1', 'BsmtFinType2',
              'BsmtFinSF1', 'BsmtFinSF2'], axis=1, inplace=True)

    return dfe, cat_encoder


def normalize(dfn, scalor=None):
    '''
    normalize the input data to N(0, 1)
    :param dfn: input data frame
    :param scalor: (median, standard deviation) tuple
    :return: dfn, scalor
    '''
    if scalor is None:
        mdn = dfn.median()
        mad = dfn.mad()
        scalor = (mdn, mad)
    dfn = (dfn - scalor[0]) / scalor[1]
    dfn.fillna(0, inplace=True)
    return dfn, scalor


class TrainXValidation:
    def __init__(self, df, fold=5):
        self.df_train = df.sample(frac=1) # shuffle all samples
        self.sample_size = len(self.df_train)
        self.xv_fold = 5  # number of cross validation

    def gen_fit_valid_set_indices(self):
        full_set = set(range(self.sample_size))
        vset_sz = self.sample_size//self.xv_fold
        tv_sets = []
        for ii in range(self.xv_fold):
            v_set = set(range(ii*vset_sz, (ii+1)*vset_sz))
            t_set = full_set.difference(v_set)
            tv_sets.append((list(t_set), list(v_set)))
        return tv_sets

    def model_fit_predict(self, dff, dfp, validate=True,  **kwargs):
        target = 'SalePrice'
        # in-sample
        tgt_f = copy.deepcopy(dff[target])
        dff, coder = encode_categoricals(dff)
        dff, scale = normalize(dff)
        features = list(set(dff.columns).difference([target]))
        #import pdb
        #pdb.set_trace()
        mdl = xgb.XGBRegressor(base_score=0.5, colsample_bylevel=0.2, colsample_bytree=0.5,
                       gamma=0.0001, learning_rate=0.001, max_delta_step=0, max_depth=3,
                       min_child_weight=1, missing=None, n_estimators=10000, nthread=2,
                       objective='reg:linear', scale_pos_weight=1, seed=0, silent=True, subsample=1,
                               **kwargs)  #reg_alpha=0, reg_lambda=10)

        mdl.fit(dff[features], dff[target])
        prd = mdl.predict(dff[features])
        cr = np.corrcoef(prd, dff[target].values)[0, 1]

        prd = prd * scale[1][target] + scale[0][target]
        rmse = np.sqrt(skl.metrics.mean_squared_error(np.log(tgt_f), np.log(prd)))
        mae = skl.metrics.mean_absolute_error(tgt_f, prd)
        print(kwargs)
        #print(mdl)
        print('in-sample: ', cr, mae, rmse)

        #valication
        if validate:
            tgt_p = copy.deepcopy(dfp[target])
        dfp, _ = encode_categoricals(dfp, coder)
        dfp, _ = normalize(dfp, scale)
        prd = mdl.predict(dfp[features])

        if validate:
            cr = np.corrcoef(prd, dfp[target].values)[0, 1]
        prd = prd * scale[1][target] + scale[0][target]
        if validate:
            rmse = np.sqrt(skl.metrics.mean_squared_error(np.log(tgt_p), np.log(prd)))
            mae = skl.metrics.mean_absolute_error(tgt_p, prd)
            print('out-sample: ', cr, mae, rmse)
            return mdl, cr, mae, rmse
        else:
            return mdl, prd

    def cross_validate(self, **kwargs):
        fv_sets = self.gen_fit_valid_set_indices()
        vmetrics = []
        for fset, vset in fv_sets:
            dff = self.df_train.iloc[fset].copy()
            dfv = self.df_train.iloc[vset].copy()
            _, cr, mae, rmse = self.model_fit_predict(dff, dfv, **kwargs)
            vmetrics.append([cr, mae, rmse])
        vmetrics = pd.DataFrame(vmetrics, columns = ['cr', 'mae', 'rmse'])
        return vmetrics

    def grid_search_model_args(self):
        metrics = dict()
        for af in [0, 0.3, 1, 3]:
            for lbd in [0, 0.03, 0.1, 0.3, 1, 3, 10]:
                metrics[(af, lbd)] = self.cross_validate(reg_alpha=af, reg_lambda=lbd)
        return pd.Panel(metrics)


def find_optimal_params_and_fit():
    df = pd.read_csv('data/train.csv', index_col=0)
    txv = TrainXValidation(df)
    mtc = txv.grid_search_model_args()
    #maes = mtc.minor_xs('mae').mean().unstack()
    lrmse = mtc.minor_xs('rmse').mean().unstack()
    af = lrmse.min().idxmin()
    lbd = lrmse.min(1).idxmin()
    print('optimial meta-parameters: alpha %f, lambda: %f'%(af, lbd))
    dft = pd.read_csv('data/test.csv', index_col=0)
    prd = txv.model_fit_predict(df, dft, False, reg_alpha=af, reg_lambda=lbd)
    #return af, lbd  # param values that minimize mae
    return prd


def main():
    df = pd.read_csv('data/train.csv', index_col=0)
    txv = TrainXValidation(df)
    mtc = txv.grid_search_model_args()
    return mtc

if __name__=='__main__':
    main()