import sys
#sys.path.append('./utils/')
import utils

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from abc import abstractmethod

from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

import lightgbm as lgb
import optuna.integration.lightgbm as lgbo


class Base_Model(object):
    @abstractmethod
    def fit(self, train_x, train_y, valid_x, valid_y):
        raise NotImplementedError
        
    @abstractmethod
    def predict(self, model, features):
        raise NotImplementedError
    
    def cv(self, train_y, train_features, test_features, fold_ids, is_reg=True): # is_reg=True:回帰, False:分類
        test_preds = np.zeros(len(test_features))
        oof_preds = np.zeros(len(train_features))
        if(is_reg==False):
            test_preds = pd.DataFrame()
            
        for i_fold, (trn_idx, val_idx) in enumerate(fold_ids):
            
            trn_x = train_features.iloc[trn_idx]
            val_x = train_features.iloc[val_idx]
            trn_y = train_y.iloc[trn_idx]
            val_y = train_y.iloc[val_idx]
            
            model = self.fit(trn_x, trn_y, val_x, val_y)
        
            oof_preds[val_idx] = self.predict(model, val_x)
            
            # 回帰
            if(is_reg):
                oof_score = np.sqrt(mean_squared_error(val_y, oof_preds[val_idx]))
                print(f'Fold{i_fold}_RMSE : {oof_score}')
                test_preds += self.predict(model, test_features)/len(fold_ids)
            # 分類
            else:
                oof_score = f1_score(val_y, np.round(oof_preds[val_idx]), average='macro') # 予測確率をroundで四捨五入
                print(f'Fold{i_fold}_F1 : {oof_score}')
                test_preds[f'Fold{i_fold}'] = self.predict(model, test_features)
                
        if(is_reg):
            oof_score = np.sqrt(mean_squared_error(train_y, oof_preds))
            print('-'*50)
            print(f'oof score : {oof_score}')
            print('-'*50)
        else:
            oof_score = f1_score(train_y, np.round(oof_preds), average='macro')
            print('-'*50)
            print(f'oof score : {oof_score}')
            print('-'*50)
            test_preds = test_preds.T.mode().loc[0] # mode:最頻値
        
        evals_results = {'evals_result':{
            'oof_score':oof_score,
            'n_data':len(train_features),
            'n_features':len(train_features.columns),
        }}
        
        return oof_preds, test_preds, evals_results
    
    
    
class Lgbm(Base_Model):
    def __init__(self, model_params):
        self.model_params = model_params
        self.models = []
        self.feature_cols = None
        
    def fit(self, train_x, train_y, valid_x, valid_y):
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_valid = lgb.Dataset(valid_x, valid_y)
        
        model = lgb.train(self.model_params,
                         train_set=lgb_train,
                         valid_sets=[lgb_valid],
                         valid_names=['valid'],
                         early_stopping_rounds=100,
                         num_boost_round=99999,
                         verbose_eval=False,
                         #categorical_feature=cat_col
                         )
        self.models.append(model)
        return model
    
    def predict(self, model, features):
        self.feature_cols = features.columns
        return np.argmax(model.predict(features), axis=1)
    
    def tuning(self, y, X, params, cv):
        lgbo_train = lgbo.dataset(X, y)
        
        tuner_cv = lgbo.LightGBMTunerCV(
        params,
        lgbo_train,
        num_boost_round=99999,
        verbose_eval=-1,
        folds=cv,
        #categorical_feature=cat_col
        )
        
        tuner_cv.run()
        print('----------------------------------------------')
        print('Best_score:',tuner_cv.best_score)
        print('Best_params:',tuner_cv.best_params)
        print('-----------------------------------------------')
        return tuner_cv.best_params
    
    
    def visualize_importance(self):
        feature_importance_df = pd.DataFrame()

        for i,model in enumerate(self.models):
            _df = pd.DataFrame()
            _df['feature_importance'] = model.feature_importance(importance_type='gain')
            _df['column'] = self.feature_cols
            _df['fold'] = i+1
            feature_importance_df = pd.concat([feature_importance_df,_df],axis=0,ignore_index=True)
        
        order = feature_importance_df.groupby('column').sum()[['feature_importance']].sort_values('feature_importance',ascending=False).index[:50]

        fig, ax = plt.subplots(1,1,figsize=(12, max(4, len(order) * .2)))
        sns.boxenplot(data=feature_importance_df, x='feature_importance', y='column', order=order, ax=ax, palette='viridis')
        ax.tick_params(axis='y')
        ax.grid()
        fig.tight_layout();
        return fig,ax
    
    
    def save(self, path):
        utils.Util.dump(self.model, path)
        
    def load(self, path):
        self.model = utils.Util.load(path)