import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

TARGET = 'is_listened'

class Model(object):
    """Build the model with all the reweight
    
    Parameters
    ----------
    preprocessing : Preprocessing instance
        And instance of Preprocessing with the loaded data and the created features.
    std_withdraw : float
        The variation of the random in the withdraw for the ratios.
    nbins_reweight : int
        The number of bins in the reweight.
    n_context_type : int
        The number observation to consider in the bias.
    coef_correct_1 : int
        First magic coeficient.
    coef_correct_2 : int
        Second magic coeficient.
    coef_correct_2 : int
        Third magic coeficient.
    features_4_weight: list or None
        The features for the reweight.
    n_fold_reweight : int
        The number of fold to compute the reweight.
    n_sample_reweight : int
        The number of observations in the reweight models.    
    """

    def __init__(self,
                 preprocessing,
                 std_withdraw=0.02,
                 nbins_reweight=15,
                 n_context_type=5,
                 coef_correct_1=0.4748,
                 coef_correct_2=0.4158,
                 coef_correct_3=0.0654,
                 features_4_weight=None,
                 n_fold_reweight=5,
                 n_sample_reweight=100000):

        self.preprocessing = preprocessing
        self.train = preprocessing.train
        self.test = preprocessing.test
        self.categorize_features = preprocessing.categorize_features
        self.listen_type_features = preprocessing.listen_type_features
        self.combo_features1 = preprocessing.combo_features1
        self.combo_features2 = preprocessing.combo_features2
        self.features = preprocessing.features
        self.flow_features = preprocessing.flow_features
        self.context_features = preprocessing.context_features
        
        if features_4_weight is None:
            features = preprocessing.features
            features_bis = preprocessing.features_bis
            features_count = preprocessing.features_count
            features_ratio = preprocessing.features_ratio
            features_4_test_correction = preprocessing.features_4_test_correction
            drop_features = features_bis + features_count + features_ratio
        
            correct_features = [feat + '_context_flow_count_bis' for feat in features_4_test_correction]
            features_4_weight = np.setdiff1d(features, drop_features).tolist() + correct_features
            
        self.users_in_train = self.train.user_id.unique()
        self.features_4_weight = features_4_weight
        self.std_withdraw = std_withdraw
        self.nbins_reweight = nbins_reweight
        self.n_context_type = n_context_type
        self.coef_correct_1 = coef_correct_1
        self.coef_correct_2 = coef_correct_2
        self.coef_correct_3 = coef_correct_3
        self.n_fold_reweight = n_fold_reweight
        self.n_sample_reweight = n_sample_reweight
        
    def _withdraw_information(self):
        """Widthdraw information of each observations in the train.
        """
        train = self.train
        # Withdraw informations of each observation in count/ratio
        for feature in self.categorize_features:
            train = information(train, feature, feature in self.listen_type_features, self.std_withdraw, self.context_features, self.flow_features)
        
        for feat1 in self.combo_features1:
            for feat2 in self.combo_features2:
                name = feat1 + '_' + feat2
                train = information(train, name, feat1 in self.listen_type_features, self.std_withdraw, self.context_features, self.flow_features)
        
        return train

    def _compute_weight(self):
        """Compute the weights.
        """
        n_bins, bins, = np.histogram(self.users_in_train, bins=self.nbins_reweight)
        train = self.train
        n_context_type = self.n_context_type
        coef_correct_1 = self.coef_correct_1
        coef_correct_2 = self.coef_correct_2
        coef_correct_3 = self.coef_correct_3
        
        user_id_weight = (train.user_id < bins[1])*1/(1.0*n_bins.shape[0]*n_bins[0])
        for k in range(1, n_bins.shape[0]-1):
            user_id_weight+= ((train.user_id >= bins[k])& (train.user_id < bins[k+1]))*1./(1.0*n_bins.shape[0]*n_bins[k])
        user_id_weight += ((train.user_id >= bins[k+1])& (train.user_id <= bins[k+2]))*1./(1.0*n_bins.shape[0]*n_bins[k+1])
        
        train['user_id_weight'] = user_id_weight/(1.0*train.groupby(['user_id'])['user_id'].transform('count'))
        
        # Third bias    
        condition = train['user_id_context_flow_count'] < n_context_type
        train.loc[condition, 'user_id_weight'] *= coef_correct_1 / (1.0*train.loc[condition, 'user_id_weight'].sum())
        train.loc[~condition, 'user_id_weight']*= (1.-coef_correct_1) / (1.0*train.loc[~condition, 'user_id_weight'].sum())
        
        # Poids Yoyo
        condition = train['user_id_context_flow_ratio'] == 0
        train.loc[condition, 'user_id_weight'] *= coef_correct_2 / max((1.0*train.loc[condition, 'user_id_weight'].sum()), 1)
        condition = train['user_id_context_flow_ratio'] == 1
        train.loc[condition, 'user_id_weight'] *= coef_correct_3 / max((1.0*train.loc[condition, 'user_id_weight'].sum()), 1)
        condition = (train['user_id_context_flow_ratio'] != 1) & (train['user_id_context_flow_ratio'] != 0)
        train.loc[condition, 'user_id_weight'] *= (1. - coef_correct_2 - coef_correct_3) / max((1.0*train.loc[condition,'user_id_weight'].sum()), 1)


    def _make_weights_4_bias(self):
        """Compute the weights 2.
        """
        train = self.train
        test = self.test
        n_folds = self.n_fold_reweight
        n_sample = self.n_sample_reweight
        features_4_weight = self.features_4_weight
        
        users_in_train = self.users_in_train
        train['label'] = 0
        test['label'] = 1
    
        # TODO: with sklearn
        cv_user_train_split = np.random.random_integers(low=0, high=n_folds-1, size=len(users_in_train))
        v_weights = np.zeros((train.shape[0], ))
    
        for i in range(n_folds):
            user_in_bag = users_in_train[cv_user_train_split != i]
            user_out_bag = users_in_train[cv_user_train_split == i]
                
            in_bag_0 = (train.loc[train.user_id.isin(user_in_bag)]).sample(n_sample, weights='user_id_weight', replace=True)
            in_bag_1 = (test.loc[test.user_id.isin(user_in_bag)]).sample(n_sample, replace=True)
            in_bag = pd.concat([in_bag_0, in_bag_1], axis=0).iloc[np.random.permutation(in_bag_0.shape[0] + in_bag_1.shape[0])]
    
            out_bag_0 = (train.loc[train.user_id.isin(user_out_bag)]).sample(n_sample, weights='user_id_weight', replace=True)
            out_bag_1 = (test.loc[test.user_id.isin(user_out_bag)]).sample(n_sample, replace=True)
            out_bag = pd.concat([out_bag_0, out_bag_1], axis=0).iloc[np.random.permutation(out_bag_0.shape[0] + out_bag_1.shape[0])]
    
                
            X_in_bag_D = xgb.DMatrix(in_bag[features_4_weight], in_bag.label)
            X_out_bag_D = xgb.DMatrix(out_bag[features_4_weight], out_bag.label)
            watchlist = [(X_in_bag_D, 'train'), (X_out_bag_D, 'eval')]
    
            params = {
                "objective": "binary:logistic",
                "booster" : "gbtree",
                "eval_metric": "auc",
                "subsample": 0.7,
                "colsample_bytree": 1,
                "colsample_bylevel" : 0.4, 
                "max_depth": 3,
                "silent": 1,
                "n_estimators":10000,
                "learning_rate":0.2,
                "reg_alpha":0,
                "reg_lambda ":1
                }            
                
            gbm = xgb.train(params, X_in_bag_D, num_boost_round=10000, evals=watchlist, early_stopping_rounds=30, verbose_eval=False)
            index_out_of_bag = train.user_id.isin(user_out_bag)
            v_weights[index_out_of_bag.values] += gbm.predict(xgb.DMatrix(train.loc[index_out_of_bag, features_4_weight]))
            
        return v_weights


    def fit(self, xgb_params, n_sample_fit=500000, test_size=0.25, 
            n_rounds=500, with_reweight=True, verbose=True, return_train_test=False):
        """Fit the modlel
        
        Parameters
        ----------
        xgb_params : dict,
            The xgboost parameters.
        n_sample_fit : int
            The number of sample for the model.
        test_size : float
            The test size.
        n_rounds : int
            The maximum number of rounds for xgboost.
        verbose : int or bool,
            The verbose of xgboost.
        return_train_test : bool
            If true, the train/test for the model building is returned.
            
        Returns
        -------
        gbm : Booster instance
            The xgboost model.
        auc : float
            The validation auc.           
        
        """
        self.train = self.preprocessing.train.copy()
        features = self.features
        
        # Withdraw observations for the liste_type
        train = self._withdraw_information()
                
        # Weight
        self._compute_weight()
        
        if with_reweight:
            train['user_id_weight'] *= self._make_weights_4_bias()
        
        user_id_train, user_id_test = train_test_split(self.users_in_train, test_size=test_size)
        train_local = train[train.user_id.isin(user_id_train)].copy()
        test_local = train[train.user_id.isin(user_id_test)].copy()
        
        train_sample = train_local.sample(n_sample_fit, weights='user_id_weight', replace=True)
        X_train = train_sample[features]
        y_train = train_sample[TARGET]
        
        n_test = int(n_sample_fit*test_size / (1. - test_size))
        test_sample = test_local.sample(n_test, weights='user_id_weight', replace=True)
        X_test = test_sample[features]
        y_test = test_sample[TARGET]
        
        test_sample_2 = test_local.sample(n_test, weights='user_id_weight', replace=True)
        X_valid = test_sample_2[features]
        y_valid = test_sample_2[TARGET]
        
        eval_set = [(X_test, y_test)]      
        model = xgb.XGBClassifier(**xgb_params)  
        clf = model.fit(X_train, y_train, eval_metric="auc", eval_set=eval_set, early_stopping_rounds=30, verbose=verbose)
        
        y_pred = clf.predict_proba(X_valid)[:, 1]
        acc = accuracy_score(y_valid, y_pred > 0.5)
        auc = roc_auc_score(y_valid, y_pred)
        
        if verbose:
            print("Accuracy: {0}".format(acc))
            print("AUC ROC: {0}".format(auc))
        
        if return_train_test:
            return clf, auc, train_sample, test_sample
        else:
            return clf, auc


def information(train, name, is_listen_type_feature, std, context_features, flow_features):
    """Withdraw the informations
    
    Parameters
    ----------
    train : DataFrame
        The train data.
    name : str
        The name of the feature to create
    is_listen_type_feature : bool
        If the flow features are modified
    std : float
        The variation of the random in the withdraw for the ratios.        
    
    """
    N = train.shape[0]
    feat_ratio = name + '_ratio'
    feat_count = name + '_count'
    train[feat_ratio] = (train[feat_ratio]*train[feat_count] - train.is_listened+np.random.binomial(1, .5, N))/(1.0*(train[feat_count] - 1).replace(0, 1))
    train[feat_ratio] += np.random.uniform(-std, std, N)
    train[feat_count] -= 1

    if context_features or not is_listen_type_feature:
        feat_context_ratio = name + '_context_ratio'
        feat_context_count = name + '_context_count'
        train[feat_context_ratio] = (train[feat_context_ratio]*train[feat_context_count] - train.is_listened_context+np.random.binomial(1, .5, N))/(1.0*(train[feat_context_count] - train['is_context']).replace(0,1))
        train[feat_context_ratio] += np.random.uniform(-std, std, N)
        train[feat_context_count] -= train['is_context']

    if is_listen_type_feature:
        if flow_features:
            feat_listen_ratio = name + '_flow_ratio'
            feat_listen_count = name + '_flow_count'
            train[feat_listen_ratio] = (train[feat_listen_ratio]*train[feat_listen_count]-train.is_listened_flow)/(1.0*(train[feat_listen_count]-train['listen_type']).replace(0, 1))
            train[feat_listen_ratio] += np.random.uniform(-std, std, N)
            train[feat_listen_count] -= train['listen_type']

        feat_listen_ratio = name + '_context_flow_ratio'
        feat_listen_count = name + '_context_flow_count'
        train[feat_listen_ratio] = (train[feat_listen_ratio]*train[feat_listen_count]-train.is_listened_context_flow)/(1.0*(train[feat_listen_count]-train['is_context_flow']).replace(0, 1))
        train[feat_listen_ratio] += np.random.uniform(-std, std, N)
        train[feat_listen_count] -= train['is_context_flow']

    return train