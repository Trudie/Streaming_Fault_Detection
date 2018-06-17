import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import implicit


class fraud_detection():
    def __init__(self):
        pass
        
    def load_stream(self, **kwargs):
        path = kwargs['params']['path']        
        streams = pd.read_json(path, lines=True)
        streams = streams.head(100000)
        streams['timestamp'] = pd.to_datetime(streams['timestamp'],
                                           format='%Y-%m-%d %H:%M:%S')
        streams.sort_values(by=['timestamp'], inplace=True)
        streams.reset_index(inplace=True, drop=True)
        streams['delta'] = streams.groupby(['user_id'])['timestamp'].diff().dt.total_seconds()
        return streams

    def load_users(self, **kwargs):
        path = kwargs['params']['path']
        users = pd.read_json(path, lines=True)
        return users
        
    def label(self, **kwargs):
        ti = kwargs['ti']
        streams = ti.xcom_pull(task_ids='load_stream')
        ab_user = []

        # cyclic
        threshold = 3

        user_play = streams.groupby('user_id').size()
        user_delta = streams.groupby('user_id')['delta'].std()
        outlier = user_play[user_play > threshold].index
        same_delta = user_delta[user_delta == 0].index
        outlier = streams.loc[streams['user_id'].isin(outlier & same_delta)]

        outlier_userId = outlier['user_id'].unique()
        ab_user = np.append(ab_user, outlier_userId)

        # play same track
        threshold = 3

        play_same = streams.groupby(['user_id', 'timestamp'])['track_id'].value_counts()
        outlier = play_same.loc[play_same > threshold]
        outlier.name = 'cnt'
        outlier = outlier.reset_index()

        outlier_userId = outlier['user_id'].unique()
        ab_user = np.append(ab_user, outlier_userId)

        # play times
        threshold = 5

        play_time = streams.groupby(['user_id', 'timestamp']).size()
        outlier = play_time[play_time > threshold].reset_index()

        outlier_userId = outlier['user_id'].unique()
        ab_user = np.append(ab_user, outlier_userId)
        return ab_user

    def build_model(self, **kwargs):
        ti = kwargs['ti']
        streams = ti.xcom_pull(task_ids='load_stream')
        users = ti.xcom_pull(task_ids='load_users')
        ab_user = ti.xcom_pull(task_ids='label')
        
        # preprocessing
        users['abnormal'] = [0] * users.shape[0]
        users['birth_year'].loc[users['birth_year'] == ''] = users['birth_year'].value_counts().index[0]
        users['birth_year'] = users['birth_year'].astype('int')
        for i in ['access', 'gender']:
            le = preprocessing.LabelEncoder()
            users[i] = le.fit_transform(users[i])

        # feature engineering
        vectorizer = CountVectorizer()
        streams = streams.groupby('user_id')['track_id'].apply(lambda x: ' '.join(x))
        counts = vectorizer.fit_transform(streams)

        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(counts)

        als = implicit.als.AlternatingLeastSquares(factors=50)
        als.fit(tfidf.T)  # item_user matrix
        user_factors = pd.DataFrame(data=als.user_factors)

        users = users.sort_values(by='user_id')
        users.reset_index(drop=True, inplace=True)
        user_factors.reset_index(drop=True, inplace=True)
        users = pd.concat([users, user_factors], axis=1, ignore_index=True)
        users.rename(columns={0: 'access', 1: 'birth_year', 2: 'country', 3: 'gender', 4: 'user_id', 5: 'abnormal'},
                     inplace=True)

        # set X, y
        y = users['abnormal']
        X = users.drop(['abnormal', 'user_id'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

        # training 
        clf = RandomForestClassifier(n_jobs=10)
        clf.fit(X_train, y_train)
        clf.score(X_test, y_test)

        # testing
        y_pred = clf.predict(X_test)
        print(confusion_matrix(y_test, y_pred))

