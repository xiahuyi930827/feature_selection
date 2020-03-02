# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 16:24:19 2019

@author: 12291
"""
#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=1978)


from sklearn.svm import SVC
import pandas as pd
import numpy as np
from pandas import DataFrame
import logging
import traceback
import time


from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif,SelectPercentile
from sklearn.model_selection import train_test_split
#移除低方差的特征 
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
#过滤式--filter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#包裹式 --wrapped
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_boston
#集成式--Embedded
from sklearn.svm import LinearSVC

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
#基于树的特征选择
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
class Feature_selection():
    '''
    输入数据类型：dataframe
    输出数据类型：dataframe 
    输入数据特征：包含特征和类别
    '''
    def __init__(self,df1,df2,var_tre):
        #self.shape=object.shape 
        self.X_feature = df1
        self.Y_label   = df2
        self.X_columns_names=df1.columns.values.tolist()
        self.var_tre=var_tre
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( df1, df2, test_size=0.33, random_state=42)

    def VarianceThreshold_selection(self):
        try:
            sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
            index_X_feature=sel.fit(self.X_feature).get_support(indices=True)
            #new_X_feature=fill_to_Dateframe(index_X_feature)
            new_X_feature=self.X_feature.iloc[:,index_X_feature]
            #sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
            #low_number_X=sel.fit_transform(self.X_feature)
            #result=DataFrame(np.c_[self.X_feature,self.Y_label],index=range(1,len(self.Y_label)+1),columns=self.columns)
        except Exception as e:
            logging.error(e)
            error_message = traceback.format_exc()
            logging.error(error_message)
            error_text = 'VarianceThreshold_selection failed!!! Time：',time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.error(error_text) 
        else:
            return new_X_feature

    #选择保留特征的个数
    def filter_SelectKBest_selection(self):
        '''k为需要调节的特征个数'''
        try:
            estimator=('selectpercentile',SelectPercentile(f_classif))
            pipe = Pipeline([self.var_tre,estimator,('preprocessing', StandardScaler()), ('classifier', SVC())])
            param_grid =[{'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
            'selectpercentile__percentile':range(1,10),
            'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},
            {'classifier': [RandomForestClassifier(n_estimators=100)],
            'selectpercentile':[None],
            'preprocessing': [None], 'classifier__max_features': range(1,10)}]
            grid = GridSearchCV(pipe, param_grid, cv=10,iid=True)
            grid.fit(X_train, y_train)
            print("Best params:\n{}\n".format(grid.best_params_))
            print("Best cross-validation score: {:.2f}".format(grid.best_score_))
            print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))          
        except Exception as e:
            logging.error(e)
            error_message = traceback.format_exc()
            logging.error(error_message)
            error_text = 'filter_SelectKBest_selection failed!!! Time：',time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.error(error_text)
        else:
            #print(type(new_X_feature),type(self.Y_label))
            return grid.best_params_#new_X_feature

    def embedded_Xgbt_selection(self):
        try:
            pipe=Pipeline([('classifier',XGBClassifier())])
            param_grid=[{
            'classifier__learning_rate':[0.01,0.02,0.05,0.1,0.3],
            'classifier__n_estimators':range(100,1500,100),
            'classifier__max_depth':range(3,10),
            'classifier__min_child_weight':range(1,12),#叶节点权重和小于该节点，则拆分结束越大越能防止过拟合
            'classifier__gamma':[i/10.0 for i in range(0,5)],#最低的损失值
            'classifier__subsample':[i/10.0 for i in range(6,10)],#选取的子样本比例
            'classifier__colsample_bytree':[i/10.0 for i in range(5,10)],#在建立树时对特征随机采样的比例
            'classifier__reg_alpha':[1e-5, 1e-2, 0.001, 0.005, 0.01, 0.05, 1, 100]
            }]
            grid = GridSearchCV(pipe, param_grid,scoring='roc_auc',cv=10,iid=True)            
            grid.fit(X_train, y_train)
            print("Best params:\n{}\n".format(grid.best_params_))
            print("Best cross-validation score: {:.2f}".format(grid.best_score_))
            print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))
        except Exception as e:
            logging.error(e)
            error_message = traceback.format_exc()
            logging.error(error_message)
            error_text = 'embedded_Xgbt_selection failed!!! Time：',time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.error(error_text)
        else:
            #print(type(new_X_feature),type(self.Y_label))
            return (grid.best_params_)#new_X_feature
        
    def embedded_GBDT_selection(self):
        try:
            pipe=Pipeline([('classifier',GradientBoostingClassifier())])
            param_grid=[{
            'classifier__learning_rate':[0.01,0.02,0.05,0.1,0.3],                    
            'classifier__n_estimators':range(20,81,20),
            'classifier__max_depth':range(3,10),
            'classifier__min_samples_split':range(100,1001,100),
            'classifier__min_samples_leaf':range(20,71,10),
            'classifier__max_features':range(7,20,2),
            'classifier__subsample':[i/10.0 for i in range(6,10)],
            'classifier__subsample':[0.6,0.7,0.75,0.8,0.85,0.9]            
            }]
            grid = GridSearchCV(pipe, param_grid,scoring='roc_auc',cv=10,iid=True)           
            grid.fit(self.X_train, self.y_train.values.ravel())
            print("Best params:\n{}\n".format(grid.best_params_))
            print("Best cross-validation score: {:.2f}".format(grid.best_score_))
            print("Test-set score: {:.2f}".format(grid.score(X_test, y_test)))
        except Exception as e:
            logging.error(e)
            error_message = traceback.format_exc()
            logging.error(error_message)
            error_text = 'embedded_Xgbt_selection failed!!! Time：',time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.error(error_text)
        else:
            #print(type(new_X_feature),type(self.Y_label))
            return (grid.best_params_)#new_X_feature
            
            
    def embedded_RandomForestRegressor_selection(self):
        try:
            rf = RandomForestRegressor(max_features = "auto",n_estimators=200,random_state = 0)
            trans_X=np.array(self.X_feature)
            trans_Y=np.array(self.Y_label.values.ravel())
            rf.fit(trans_X,trans_Y)
            importances=rf.feature_importances_
            sorted_importances=sorted(importances,reverse=True)
            indices=self.X_feature.iloc[:,np.argsort(importances)[::-1]]
            #rf.oob_score_
        except Exception as e:
            logging.error(e)
            error_message = traceback.format_exc()
            logging.error(error_message)
            error_text = 'RandomForestRegressor_selection failed!!! Time：',time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.error(error_text) 
        else:
            return (sorted_importances,indices)
       
    #线性回归剔除特征个数
    def wrapped_LinearRegression_selection(self):
        '''n_features_to_select 选择剔除特征的个数'''
        try:
            lr=LinearRegression()
            rfe=RFE(lr,n_features_to_select=2)#选择保留的个数
            rfe.fit(self.X_feature,self.Y_label.values.ravel())
            #importances=ref.feature_importances_
            new_X_feature=self.X_feature.iloc[:,rfe.support_]
            #result=sorted(zip(map(lambda x:round(x,4), rfe.ranking_),self.X_columns_names))
        except Exception as e:
            logging.error(e)
            error_message = traceback.format_exc()
            logging.error(error_message)
            error_text = 'wrapper_selection failed!!! Time：',time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.error(error_text)
        else:
            return new_X_feature

    def wrapped_LogisticRegression_selection(self):
        '''n_features_to_select 选择剔除特征的个数'''
        try:          
            lr=LogisticRegression(max_iter=10000,solver='lbfgs',multi_class='auto')
            estimator=('lr',RFE(lr))
            #rfe=RFE(lr,n_features_to_select=3)#选择剔除1个
            pipe = Pipeline([self.var_tre,estimator])
            param_grid=[{'lr__n_features_to_select':range(1,15)}]
            grid = GridSearchCV(pipe, param_grid, cv=10,iid=True)
            grid.fit(X_train, y_train)
            print("Best params:\n{}\n".format(grid.best_params_))
            print("Best cross-validation score: {:.2f}".format(grid.best_score_))
            print("Test-set score: {:.2f}".format(grid.score(X_test, y_test))) 
        except Exception as e:
            logging.error(e)
            error_message = traceback.format_exc()
            logging.error(error_message)
            error_text = 'wrapper_selection failed!!! Time：',time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.error(error_text)
        else:
            return grid.best_params_

    def embedded_LinearSVC_selection(self):
        '''基于L1的特征选择'''
        try:
            clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC(max_iter=5000,C=0.01,penalty='l1',dual=False))),
            ('classification', RandomForestClassifier())])
            param_grid=[{'classification__n_estimators':list(range(100,1000,10))}]
            grid = GridSearchCV(clf, param_grid, cv=10,iid=True)
            grid.fit(X_train, y_train)
            print("Best params:\n{}\n".format(grid.best_params_))
            print("Best cross-validation score: {:.2f}".format(grid.best_score_))
            print("Test-set score: {:.2f}".format(grid.score(X_test, y_test))) 
            #new_X_feature=model.transform(self.X_feature)
            #new_X_feature=self.X_feature.iloc[:,model.get_support(indices=False)]
            #a=model.feature_importances
        except Exception as e:
            logging.error(e)
            error_message = traceback.format_exc()
            logging.error(error_message)
            error_text = 'wrapper_selection failed!!! Time：',time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.error(error_text)
        else:
            return grid.best_params_
        
    def embedded_ExtraTreesClassifier_selection(self):
        try:
            clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC(max_iter=5000,C=0.01,penalty='l1',dual=False))),
            ('classification', RandomForestClassifier())])
            param_grid=[{
            'feature_selection__threshold':[0.01,0.02],
            'classification':[RandomForestClassifier()],
            'classification__n_estimators':range(100,200,10),
            'feature_selection__threshold':[0.01,0.02],
             'classification':[ExtraTreesClassifier()],
            'classification__n_estimators':range(100,200,10)}]
            grid = GridSearchCV(clf, param_grid, cv=10,iid=True)
            grid.fit(X_train, y_train)
            print("Best params:\n{}\n".format(grid.best_params_))
            print("Best cross-validation score: {:.2f}".format(grid.best_score_))
            print("Test-set score: {:.2f}".format(grid.score(X_test, y_test))) 
        except Exception as e:
            logging.error(e)
            error_message = traceback.format_exc()
            logging.error(error_message)
            error_text = 'wrapper_selection failed!!! Time：',time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.error(error_text)
        else:
            return grid.best_params_
        
    def pipeline_selection(self,estimator,parameters):
        try: #去除低方差特性
            pip=Pipeline([self.var_tre,estimator])
            grid = GridSearchCV(pip,param_grid = parameters,cv=5,iid=True) #这里的scoring需要自己设置
            grid.fit(self.X_train, self.y_train.values.ravel())
            print(grid.score(self.X_test,self.y_test))
            model = grid.best_estimator_
            yfit = model.predict(self.X_test)
            print(classification_report(self.y_test, yfit, target_names=['0','1','2']))
        except Exception as e:
            logging.error(e)
            error_message = traceback.format_exc()
            logging.error(error_message)
            error_text = 'wrapper_selection failed!!! Time：',time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.error(error_text)
        else:
            return grid.best_params_ 
         
            
if __name__ == '__main__':  
    iris = load_breast_cancer()
    X = iris["data"]
    Y = iris["target"]
    names = iris["feature_names"]
    column_X=DataFrame(X,index=range(len(X)),columns=names)
    column_Y=DataFrame(Y,index=range(len(X)),columns=['label'])
    estimator1=[('RFC',RandomForestClassifier()),('RFR',RandomForestRegressor()),('ETC',ExtraTreesClassifier())]
    estimator2=[('SVC',SVC()),('LSVC',LinearSVC()),('LR',LinearRegression())]
    estimator=estimator2[0]
    var_tre=('Var',VarianceThreshold(threshold=(.8 * (1 - .8))))
    parameters={'SVC__C':[0.001,0.01,1,10,100],'SVC__gamma':[0.001,0.01,1,10,100],'SVC__decision_function_shape':['auto']}
    fs=Feature_selection(column_X,column_Y,var_tre)
    a=fs.embedded_GBDT_selection()
    print(a)
 
    '''