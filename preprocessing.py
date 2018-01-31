import pandas as pd
import numpy as np

from loading import load_train, load_test

CONTEXT_TYPE_TEST = [1, 5, 20, 23]

class Preprocessing():
    """Load, clean and creature new features.
    
    Parameters
    ----------
    categorize_features : list of str
        The features to compute the counts and ratios.
    listen_type_features : list of str
        The features to compute the flow counts and ratios.
    combo_features1 : list of str
        The 1st feature of the conditional counts and ratios.
    combo_features2 : list of str
        The 2nd feature of the conditional counts and ratios.
    features_4_test_correction : list of str
        The features to correct in the test.
    fillna : bool
        If we fill the NaN values.
    """
    def __init__(self, 
                 categorize_features=['user_id', 'artist_id', 'media_id', 'genre_id'], 
                 listen_type_features=['user_id', 'media_id', 'media_id', 'genre_id'], 
                 combo_features1=['user_id'],
                 combo_features2=['genre_id', 'artist_id', 'platform_family'], 
                 features_4_test_correction=['user_id', 'artist_id', 'media_id', 'genre_id'],
                 drop_features=['album_id'],
                 flow_features=False,
                 context_features=False,
                 fillna=False):

        self.categorize_features = categorize_features
        self.listen_type_features = listen_type_features
        self.combo_features1 = combo_features1
        self.combo_features2 = combo_features2
        self.features_4_test_correction = features_4_test_correction
        self.drop_features = drop_features
        self.flow_features = flow_features
        self.context_features = context_features
        self.fillna = fillna
        
    @property
    def features_bis(self):
        """The bis features
        """
        features_bis = []
        for feat in self.train.columns:
            if '_bis' in feat:
                features_bis.append(feat)
                
        return features_bis
    
    @property
    def features_count(self):
        """The count features.
        """
        features_count = []
        for feat in self.features:
            if '_count' in feat:
                features_count.append(feat)
                
        return features_count
    
    
    @property
    def features_ratio(self):
        """The raio features.
        """
        features_ratio = []
        for feat in self.features:
            if '_ratio' in feat:
                features_ratio.append(feat)
                
        return features_ratio
    
    @property
    def features(self):
        """The features used in the prediction model.
        """
        other_features = ['listen_type', 'is_context', 'is_context_flow', 
                  'is_listened_context', 'is_listened_flow', 
                  'is_listened_context_flow']
        
        drop_features = self.categorize_features + self.drop_features + other_features + self.features_bis
        features = np.setdiff1d(self.train.columns.tolist(), drop_features + ['is_listened'], assume_unique=True)
        
        return features        
        
    def load_raw_data(self, nrows=None, save_to_hdf=False):
        """Load the data from the csv or hdf (if it exists)
        
        Parameters
        ----------
        nrows : int or None
            The number of lines to load in the train.
        save_to_hdf : bool
            If the file is saved in hdf after beeing loaded            
        """
        train = load_train(nrows=nrows, save_to_hdf=save_to_hdf)
        test = load_test(save_to_hdf=save_to_hdf)

        train.drop(self.drop_features, axis=1, inplace=True)
        test.drop(self.drop_features, axis=1, inplace=True)
        
        correct_dates(train)
        to_datetime(train, keep_dates=True)
        correct_dates(test)
        to_datetime(test, keep_dates=True)
        test.loc[test.listen_type == 0, 'listen_type'] = 1
        
        train['diff_days'] = (train.dt_listen - train.dt_media).dt.days
        test['diff_days'] = (test.dt_listen - test.dt_media).dt.days
        train.drop(['dt_listen', 'dt_media',], axis=1, inplace=True)
        test.drop(['dt_listen', 'dt_media',], axis=1, inplace=True)
        
        train.to_pickle('../input/train_clean.pkl')
        test.to_pickle('../input/test_clean.pkl')

        self.train = train
        self.test = test
                
    def load_cleaned_data(self):
        """Load the data cleaned.
        """
        try:
            self.train = pd.read_pickle('../input/train_clean.pkl')
            self.test = pd.read_pickle('../input/test_clean.pkl')
        except FileNotFoundError:
            self.load_raw_data()

    def create_new_features(self):
        """Create the new features count and ratios.
        """
        train = self.train
        
        train['is_context'] = train['context_type'].isin(CONTEXT_TYPE_TEST)
        train['is_context_flow'] = train['listen_type'] * train['is_context']
        
        train['is_listened_context'] = train['is_listened'] * train['is_context']
        train['is_listened_flow'] = train['is_listened'] * train['listen_type']
        train['is_listened_context_flow'] = train['is_listened'] * train['is_context_flow']
    
        for feature in self.categorize_features:
            gby_feat = train.groupby(feature)
            new_features(train, gby_feat, feature, feature in self.listen_type_features, self.context_features, self.flow_features, self.fillna)
    
        # Variable combinations
        for feat1 in self.combo_features1:
            for feat2 in self.combo_features2:
                gby_feat = train.groupby([feat1, feat2])
                name = feat1 + '_' + feat2
                new_features(train, gby_feat, name, feat1 in self.listen_type_features, self.context_features, self.flow_features, self.fillna)
        

    def create_time_features(self):
        """
        """
        train.sort(['user_id', 'ts_listen'], ascending=[True, True], inplace=True)
        train = train.reset_index(drop=True)

        test.sort(['user_id', 'ts_listen'], ascending=[True, True], inplace=True)
        test = test.reset_index(drop=True)
        dict_chronology_var = add_chronology_var(data_train = train[["user_id","ts_listen","listen_type","is_listened","media_duration"]],
                                        data_test = test[["user_id","ts_listen","listen_type","media_duration"]])

        train['is_last_listened'] = dict_chronology_var['train'].is_last_listened.values
        train['time_since_last_listen'] = dict_chronology_var['train'].time_since_last_listen.values

        test['is_last_listened'] = dict_chronology_var['test'].is_last_listened.values
        test['time_since_last_listen'] = dict_chronology_var['test'].time_since_last_listen.values
        

    def add_features_in_test(self, low=1, high=5, keep_full_train=False):
        """Merge the count and ratios of the user_id in the test file.
        
        Parameters
        ----------
        low : int
            Minimum count for the context in the randomization.
        high : int
            Maximum count for the context in the randomization.
        keep_full_train : bool
            If the full train is kept in the train_full attribute.
        """
        train = self.train
        test = self.test
        
        if keep_full_train:
            self.train_full = self.train.copy()
        
        for feature in self.categorize_features:
            gby_feat = train.groupby([feature])
            test = add_in_test(test, gby_feat, feature, feature in self.listen_type_features, feature, self.context_features, self.flow_features, self.fillna)
    
        for feat1 in self.combo_features1:
            for feat2 in self.combo_features2:
                gby_feat = train.groupby([feat1, feat2])
                name = feat1 + '_' + feat2
                test = add_in_test(test, gby_feat, name, feat1 in self.listen_type_features, [feat1, feat2], self.context_features, self.flow_features, self.fillna)
        
        self.train = train[(train.listen_type==1) & train.context_type.isin(CONTEXT_TYPE_TEST)].copy()
        
        # Correction
        for feat in self.features_4_test_correction:
            name = feat + '_context_flow_count_bis'
            ids = test[name].isnull() | (test[name]==0)
            test.loc[ids, name] = np.random.random_integers(low=low, high=high, size=ids.sum())
            if feat == 'genre_id':
                test.loc[(test[name] == 0), name] = 1
                
        self.test = test
        
    def save_data_pickle(self, save_full=False):
        """Save the data in a pickle file.
        
        Parameters
        ----------
        save_full : bool
            If the full train is also saved
        """
        self.train.to_pickle('../input/train_mod.pkl')
        self.test.to_pickle('../input/test_mod.pkl')
        if save_full:
            self.train_full.to_pickle('../input/train_full_mod.pkl')
        
        
    def load_data_pickle(self, load_full=False):
        """Load the data from the pickle file.
        """
        self.train = pd.read_pickle('../input/train_mod.pkl')
        self.test = pd.read_pickle('../input/test_mod.pkl')
        if load_full:
            self.train_full = pd.read_pickle('../input/train_full_mod.pkl')


def new_features(train, gby_feat, name, is_listen_type_feature, context_features, flow_features, fillna):
    """Create the count and ratio for a given groupby
    """
    
    # count and ratio on the all train
    count = gby_feat['is_listened'].transform('count')
    train[name + '_count'] = count
    train[name + '_count_bis'] = count
    train[name + '_ratio'] = gby_feat['is_listened'].transform('mean')
    
    if context_features:
        # Count and ratio for context observations
        count = gby_feat['is_context'].transform('sum')
        train[name + '_context_count'] = count
        train[name + '_context_count_bis'] = count
        train[name + '_context_ratio'] = gby_feat['is_listened_context'].transform('sum')/(1.*count)
        # Note that there should be NaN values if count=0.
        if fillna:
            train[name + '_context_ratio'].fillna(0.5, inplace=True)
    
    # Count and ration fot the flow observations
    if is_listen_type_feature:
        if flow_features:
            count = gby_feat['listen_type'].transform('sum')
            train[name + '_flow_count'] = count
            train[name + '_flow_count_bis'] = count
            train[name + '_flow_ratio'] = gby_feat['is_listened_flow'].transform('sum')/(1.*count)
            if fillna:
                train[name + '_flow_ratio'].fillna(0.5, inplace=True)
        
        count = gby_feat['is_context_flow'].transform('sum')
        train[name + '_context_flow_count'] = count
        train[name + '_context_flow_count_bis'] = count
        train[name + '_context_flow_ratio'] = gby_feat['is_listened_context_flow'].transform('sum')/(1.*count)
        if fillna:
            train[name + '_context_flow_ratio'].fillna(0.5, inplace=True)


def add_in_test(test, gby_feat, name, is_listen_type_feature, on_features, context_features, flow_features, fillna):
    """
    """
    test = test.join(gby_feat[name + '_count'].mean(), on=on_features)
    test = test.join(gby_feat[name + '_count_bis'].mean(), on=on_features)
    test = test.join(gby_feat[name + '_ratio'].mean(), on=on_features)
    if fillna:
        test[name + '_count'].fillna(0, inplace=True)
        test[name + '_count_bis'].fillna(0, inplace=True)
        test[name + '_ratio'].fillna(0.5, inplace=True)

    if context_features or not is_listen_type_feature:
        test = test.join(gby_feat[name + '_context_count'].mean(), on=on_features)
        test = test.join(gby_feat[name + '_context_count_bis'].mean(), on=on_features)
        test = test.join(gby_feat[name + '_context_ratio'].mean(), on=on_features)
        if fillna:
            test[name + '_context_count'].fillna(0, inplace=True)
            test[name + '_context_count_bis'].fillna(0, inplace=True)
            test[name + '_context_ratio'].fillna(0.5, inplace=True)

    if is_listen_type_feature:
        if flow_features:
            test = test.join(gby_feat[name + '_flow_count'].mean(), on=on_features)
            test = test.join(gby_feat[name + '_flow_count_bis'].mean(), on=on_features)
            test = test.join(gby_feat[name + '_flow_ratio'].mean(), on=on_features)
            if fillna:
                test[name + '_flow_count'].fillna(0, inplace=True)
                test[name + '_flow_count_bis'].fillna(0, inplace=True)
                test[name + '_flow_ratio'].fillna(0.5, inplace=True)
            
        test = test.join(gby_feat[name + '_context_flow_count'].mean(), on=on_features)
        test = test.join(gby_feat[name + '_context_flow_count_bis'].mean(), on=on_features)
        test = test.join(gby_feat[name + '_context_flow_ratio'].mean(), on=on_features)
        if fillna:
            test[name + '_context_flow_count'].fillna(0, inplace=True)
            test[name + '_context_flow_count_bis'].fillna(0, inplace=True)
            test[name + '_context_flow_ratio'].fillna(0.5, inplace=True)

    return test
             

def add_chronology_var(data_train, data_test):
    # on doit mettre train et test avec comme variables :
    # user_id, time_listen, is_listened (train)

    # variable is_last_listened
    data_train = train[["user_id","ts_listen","listen_type","is_listened","media_duration"]]
    data_test = test[["user_id","ts_listen","listen_type","media_duration"]]
    data_test['is_listened'] = -1
    data = pd.concat([data_train, data_test], axis=0, ignore_index=True)
    data.sort(['user_id', 'ts_listen'], ascending=[True, True], inplace=True)
    data = data.reset_index(drop=True)

    ## is_last_listened
    data['is_last_listened'] = -1
    index_test = np.where(data['is_listened'] == -1)[0]
    index_train = np.setdiff1d(np.arange(data.shape[0]), index_test)
    index_new_user = np.append(np.array([0]) , index_test[:(len(index_test)-1)] + 1)
    index_not_new_user = np.setdiff1d(np.arange(data.shape[0]), index_new_user)
    data.is_last_listened[index_not_new_user] = data.is_listened[(index_not_new_user-1)].values

    ## time_since_last_isten
    data['time_since_last_listen']= -1
    data.time_since_last_listen[index_not_new_user] = (data.ts_listen[index_not_new_user].values - 
                                                       data.ts_listen[(index_not_new_user-1)].values)

    return {'train' : data.ix[index_train, ['is_last_listened','time_since_last_listen']] ,
            'test' : data.ix[index_test, ['is_last_listened','time_since_last_listen']]}


        
def correct_dates(data, date_feature='release_date'):
    """Correct weird dates.
    """
    data.loc[data[date_feature] < 19170000, date_feature] += 1000000
    data.loc[data[date_feature] > 20180000, date_feature] -= 10000000


def to_datetime(data, keep_dates=False):
    """Categorize the dates from the data and correct the inconsistencies.

    Parameters
    ----------
    data : pd.Dataframe
        The data with the features "ts_listen" and "ts_listen".
    """
    # Date series
    dates_listen = pd.to_datetime(data['ts_listen'], unit='s')
    dates_media = pd.to_datetime(data['release_date'], format="%Y%m%d")

    # When the listen dates are before the release dates
    idx_wrong_dates = np.where(dates_listen < dates_media)[0]
    for idx in idx_wrong_dates:
        dates_listen[idx] = dates_listen[idx].replace(
            day=dates_media[idx].day, 
            month=dates_media[idx].month, 
            year=dates_media[idx].year)

    # Replace the year of the date before the creation of deezer...
    tmp = dates_listen.dt.year < 2016
    dates_listen[tmp] = dates_listen[tmp].apply(lambda dt: dt.replace(year=2016))

    # Add the new features and drop the previous ones
    data.drop(['ts_listen', 'release_date'], axis=1, inplace=True)
    data['day_listen'] = dates_listen.dt.weekday
    data['hour_listen'] = dates_listen.dt.hour
    data['year_media'] = dates_media.dt.year

    if keep_dates:
        data['dt_listen'] = dates_listen
        data['dt_media'] = dates_media