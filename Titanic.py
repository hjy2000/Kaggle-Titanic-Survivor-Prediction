# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:04:50 2020

@author: ASUS
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score,roc_auc_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from time import time

def fit_model(alg,parameters):
    X = np.array(train_x)
    y = np.array(train_y)
    scorer = make_scorer(roc_auc_score) #评分标准
    grid = GridSearchCV(alg,parameters,scoring = scorer,cv = 5)
    start = time() #计时
    grid = grid.fit(X,y)
    end = time()
    t = round(end - start,3)
    print("搜索时间：",t)
    print(grid.best_params_) #输出最佳参数
    return grid




#读取数据集

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

print('训练数据集：',train.shape)
print('测试数据集：',test.shape)

full=train.append(test,ignore_index=True)
full.info()

#从'Name'中提取称谓
def get_title(Series):
    title=Series.split(',')[1]
    title=title.split('.')[0]
    title=title.strip()
    return title
full['title']=full.Name.map(get_title)
#对称谓进行分类映射出'title'
title_mapDict={'Mr':'Mr',
               'Mrs':'Mrs',
               'Miss':'Miss',
               'Master':'Master',
               'Don':'Royalty',
               'Rev':'Officer',
               'Dr':'Officer',
               'Mme':'Mrs',
               'Ms':'Ms',
               'Major':'Officer',
               'Lady':'Royalty',
               'Sir':'Royalty',
               'Mlle':'Miss',
               'Col':'Officer',
               'Capt':'Officer',
               'the Countess':'Royalty',
               'Jonkheer':'Master',
               'Dona':'Royalty'}
full['title']=full['title'].map(title_mapDict)
full.drop('Name',axis=1,inplace=True)

#利用'Pclass'、'Sex'、'title'对'Age'构建随机森林模型进行训练。
age_Df=full[['Age','Pclass','Sex','title']]
age_Df=pd.get_dummies(age_Df)

age_train=age_Df[age_Df.Age.notnull()].values
age_test =age_Df[age_Df.Age.isnull()].values
age_train_x=age_train[:, 1:]
age_train_y=age_train[:, 0]
age_model=RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
age_model.fit(age_train_x,age_train_y)

# 预测出未知的Age
pre_age=age_model.predict(age_test[:, 1::])
full.loc[ (full.Age.isnull()), 'Age' ]=pre_age

#Cabin中元素有n个空格就代表有n+1张票是一起的
full.loc[full.Cabin.isnull(),'Cabin']='Unknown'
a=[i.count(' ') for i in list(full['Cabin'])]
full['Multi_Cabin_counts']=a
full['Fare']=full['Fare']/(full['Multi_Cabin_counts']+1)

import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x='Fare',y='Embarked',hue='Pclass',data=full)
plt.show()

full['Cabin']=full['Cabin'].map(lambda x:x[0])
table=full.pivot_table('Fare',columns='Pclass',index='Cabin',aggfunc='mean')
print(table)

print(full.loc[full.Fare.isnull()])

full.loc[full.Fare.isnull(),'Fare']=13.35

Emb_counts=full.Embarked.value_counts()
print(Emb_counts)

full['Embarked']=full['Embarked'].fillna('S') 

cabin_mapDict={'A':1,
               'B':2,
               'C':3,
               'D':4,
               'E':5,
               'F':6,
               'G':7,
               'T':8,
               'U':9}
full['Cabin']=full['Cabin'].map(lambda x:x[0])
full['Cabin_num']=full['Cabin'].map(cabin_mapDict)

#方法跟'Age'预测类似
cabin_Df=full[['Cabin_num','Embarked','Fare','Pclass']]
cabin_Df=pd.get_dummies(cabin_Df)
cabin_train=cabin_Df.loc[full.Cabin_num!=9].values
cabin_test=cabin_Df.loc[full.Cabin_num==9].values
cabin_train_x=cabin_train[:,1:]
cabin_train_y=cabin_train[:,0]
cabin_model=RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1)
cabin_model.fit(cabin_train_x,cabin_train_y)
pred_cabin=cabin_model.predict(cabin_test[:,1::])
full.loc[full.Cabin=='U','Cabin_num']=pred_cabin
cabin_mapDict_T=dict((i,j) for j,i in cabin_mapDict.items())
full['Cabin']=full['Cabin_num'].map(cabin_mapDict_T)
full.drop('Cabin_num',axis=1,inplace=True)

full.info()

Cabin_Df=pd.get_dummies(full['Cabin'],prefix='Cabin')
full=pd.concat([full,Cabin_Df],axis=1)
full.drop('Cabin',axis=1,inplace=True)

Embarked_Df=pd.get_dummies(full['Embarked'],prefix='Embarked')
full=pd.concat([full,Embarked_Df],axis=1)
full.drop('Embarked',axis=1,inplace=True)

full['Family_size']=full['Parch']+full['SibSp']+1
Family_size_Df=pd.DataFrame()
Family_size_Df['Family_single']=full['Family_size'].map(lambda x:1 if x==1 else 0)
Family_size_Df['Family_small']=full['Family_size'].map(lambda x:1 if 1<x<4 else 0)
Family_size_Df['Family_large']=full['Family_size'].map(lambda x:1 if x>3 else 0)
full=pd.concat([full,Family_size_Df],axis=1)

Pclass_Df=pd.get_dummies(full['Pclass'],prefix='Pclass')
full=pd.concat([full,Pclass_Df],axis=1)
full.drop('Pclass',axis=1,inplace=True)

sex_mapDict={'male':1,'female':0}
full['Sex']=full['Sex'].map(sex_mapDict)

title_Df=pd.get_dummies(full['title'],prefix='title')
full=pd.concat([full,title_Df],axis=1)
full.drop('title',axis=1,inplace=True)

print(full.corr()['Survived'].abs().sort_values(ascending=False))

full_x=pd.concat([title_Df,full['Sex'],Pclass_Df,full['Fare'],Family_size_Df,Embarked_Df,Cabin_Df,full['Age']],axis=1)
'''
full_x=pd.concat([full['title_Mr'],full['Sex']                  
,full['title_Mrs']            
,full['title_Miss']           
,full['Pclass_3']            
,full['Pclass_1']              
,full['Fare']                  
,full['Family_small']         
,full['Family_single']],axis=1)
'''
source_x=full_x.loc[:890,:]
source_y=full.loc[:890,'Survived']
pred_x=full_x.loc[891:,:]
train_x,test_x,train_y,test_y=train_test_split(source_x,source_y,train_size=0.8)

model=RandomForestClassifier(random_state = 10,warm_start = True,n_estimators = 26,max_depth = 6,max_features = 'sqrt')
model=fit_model(model,{'random_state':range(5,50,5),'warm_start':[True,False],'n_estimators':range(10,100,10),'max_depth':range(1,10,1),'max_features':['sqrt']})
model.fit(train_x,train_y)

#预测存活率
pred_y=model.predict(pred_x)
PassengerId=full.loc[891:,'PassengerId']
Submit=pd.DataFrame({'PassengerId':PassengerId,
                     'Survived':pred_y}) 

#结果
#Submit.to_csv('try(2).csv')