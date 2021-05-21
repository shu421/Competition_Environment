class Base_Model(object):
    @abstractmethod
    def fit(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError

    @abstractmethod
    def predict(self, model, features):
        raise NotImplementedError

    def cv(self, y_train, train_features, test_features, fold_ids,is_rmsle=True):
        test_preds = np.zeros(len(test_features))
        oof_preds = np.zeros(len(train_features))
        if(is_rmsle==False):
            test_preds = pd.DataFrame()

        for i_fold, (trn_idx, val_idx) in enumerate(fold_ids):

            x_trn = train_features.iloc[trn_idx]
            y_trn = y_train[trn_idx]
            x_val = train_features.iloc[val_idx]
            y_val = y_train[val_idx]

            model = self.fit(x_trn, y_trn, x_val, y_val)

            oof_preds[val_idx] = self.predict(model, x_val)
            if(is_rmsle):
                oof_score = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
                print('fold{}:RMSLE {}'.format(i_fold,oof_score))
                test_preds += self.predict(model, test_features) / len(fold_ids)
            else:
                oof_score = f1_score(y_val,np.round(oof_preds[val_idx]),average='macro')
                print('fold{}:F1 {}'.format(i_fold,oof_score))
                test_preds['fold_{}'.format(i_fold)] = self.predict(model, test_features)


        if(is_rmsle):
            oof_score = np.sqrt(mean_squared_error(y_train, oof_preds))
            print('------------------------------------------------------')
            print(f'oof score: {oof_score}')
            print('------------------------------------------------------')

        else:
            oof_score = f1_score(y_train,np.round(oof_preds),average='macro')
            print('------------------------------------------------------')
            print(f'oof score: {oof_score}')
            print('------------------------------------------------------')
            test_preds = test_preds.T.mode().loc[0]

        evals_results = {"evals_result": {
            "oof_score": oof_score,
            "n_data": len(train_features),
            "n_features": len(train_features.columns),
        }}

        return oof_preds, test_preds, evals_results

class Lgbm(Base_Model):
    def __init__(self,model_params):
        self.model_params = model_params
        self.models = []
        self.feature_cols = None
        
    def fit(self,x_train,y_train,x_valid,y_valid):
        lgb_train = lgb.Dataset(x_train,y_train)
        lgb_valid = lgb.Dataset(x_valid,y_valid)
        
        model = lgb.train(self.model_params,
            train_set=lgb_train,
            valid_sets=[lgb_valid],
            valid_names=['valid'],
            early_stopping_rounds=20,
            num_boost_round=10000,
            verbose_eval=False,
            categorical_feature=cat_col)
        self.models.append(model)
        return model
    
    def predict(self,model,features):
        self.feature_cols = features.columns
        return np.argmax(model.predict(features),axis=1)

    def tuning(self,y,X,params,cv):
        lgbo_train = lgbo.Dataset(X, y)

        tuner_cv = lgbo.LightGBMTunerCV(
        params, lgbo_train,
        num_boost_round=10000,
        early_stopping_rounds=20,
        verbose_eval=100,
        folds=cv,
        categorical_feature=cat_col
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

        fig, ax = plt.subplots(2,1,figsize=(max(6, len(order) * .4), 14))
        sns.boxenplot(data=feature_importance_df, x='column', y='feature_importance', order=order, ax=ax[0], palette='viridis')
        ax[0].tick_params(axis='x', rotation=90)
        ax[0].grid()
        fig.tight_layout()
        return fig,ax