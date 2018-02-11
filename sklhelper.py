# This module provides a simple and convenient interface for performing a ranked
# assessment of several sci-kit learn predictors based on a k-fold validation test.

#--------------------------------------------
# data handling
import pandas as pd

# iterating
import itertools

# splitting data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# evaluating accuracy
from sklearn.metrics import accuracy_score
#--------------------------------------------
# classifiers
from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import xgboost
#--------------------------------------------

class sklhelpClassify:

    def __init__(self):

        self.data = None
        self.target = None
        #--------------------------------------------
        # classifier parameters
        self.rf_params = {'n_estimators' : 75,
                          'min_samples_leaf' : 10}

        self.et_params = {'n_estimators' : 75,
                          'min_samples_leaf' : 10}
        self.knn_params  =  {'n_neighbors' : 5,
                'weights' : 'uniform',
                'algorithm' : 'ball_tree',
                'leaf_size' : 30}

        self.mlpc_params = {'activation':'identity',
               'solver':'adam',
               'shuffle':True}

        self.svc_params  =  {'C' : 1,
                'kernel' : 'rbf',
                'gamma' : 'auto'}

        self.lsvc_params = {'dual' : False}

        self.gbc_params = {}

        self.logreg_params = {}

        self.gnbayes_params = {}
        #--------------------------------------------


        # model instances
        self.models = models = {
        'Random Forest' : RandomForestClassifier(**self.rf_params),
        'Extra Trees' : ExtraTreesClassifier(**self.et_params),
        'Gaussian Naive Bayes' : GaussianNB(),
        'Logistic Regression' : LogisticRegression(),
        'Perceptron' : Perceptron(),
        'Stochastic Gradient Descent' : SGDClassifier(),
        'Support Vector Classifier' : SVC(**self.svc_params),
        'Linear SVC' : LinearSVC(**self.lsvc_params),
        'k-Nearest Neigbors' : KNeighborsClassifier(**self.knn_params),
        'Decision Tree' : DecisionTreeClassifier(),
        'Adaptive Boost Classifier' : AdaBoostClassifier(),
        'Gradient Boosting Classifier' : GradientBoostingClassifier(),
        'eXtreme Gradient Boosting' : xgboost.XGBClassifier(),
        'Multilayer Perceptron' : MLPClassifier(**self.mlpc_params),
        }

        # model keys
        self.keys = list(self.models.keys())

        # initalize dataframe to hold the accuracy of each test
        self.kf_data = pd.DataFrame(self.keys, columns= ['model'])

    # import the data as a pandas DataFrame
    def get_data(self,df):
        self.data = df

    # chose column name for predicted value
    def set_target(self,name):
        self.target = name

    # run the k-fold test
    def kfold(self,num_folds=5):

        ## initialize folds
        kf = KFold(n_splits=num_folds)

        ## split data into folds
        folds = list(kf.split(self.data))

        ## loop through models, folds
        for key, n in itertools.product(self.keys, range(len(folds))):

            ## the training data is in the first entry of the fold
            x_train = self.data.iloc[folds[n][0]].drop([self.target],axis=1)
            y_train = self.data.iloc[folds[n][0]][self.target]

            ## the testing data is in the second entry of the fold
            x_test = self.data.iloc[folds[n][1]].drop([self.target],axis=1)
            y_test = self.data.iloc[folds[n][1]][self.target]

            ## train the models
            self.models[key].fit(x_train, y_train)

            ## make the prediction
            y_prediction = self.models[key].predict(x_test)

            ## store the accuracy score
            self.kf_data.loc[self.kf_data.model == key, str(n)] = round(accuracy_score(y_prediction, y_test) * 100, 2)



    # display options

    # ranked summary
    def ranked_summary(self):
        ## compute the stats
        self.kf_data['mean'] = self.kf_data.mean(axis=1)
        self.kf_data['median'] = self.kf_data.loc[:, self.kf_data.columns != 'mean'].median(axis=1)
        self.kf_data['std_dev'] = self.kf_data.loc[:, ((self.kf_data.columns != 'mean')
                                             & (self.kf_data.columns != 'median'))].std(axis=1)
        ## display
        print(self.kf_data[['model', 'mean', 'median', 'std_dev']].sort_values(by = ['mean'], ascending=0))

    # full report
    def report(self):
        print(self.kf_data)
