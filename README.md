# pro_mobilepricepredict_ML
手机价格预测
    任务：建立不同的集成学习模型进行手机价格等级预测，并使用交叉验证选择最优的超参数
    数据集来源： https://www.kaggle.com/vikramb/mobile-price-eda-prediction
    
    
     
使用的模型及相关参数配置。该项目中使用了8个机器学习模型，其中后4个模型为集成学习模型，并为不同的学习
模型指定了参数空间。 如：RF，指定了超参数的名称n_estimators 及其搜索的空间[100, 150, 200, 250]

def main():
...
    sclf = StackingClassifier(classifiers=[KNeighborsClassifier(),
                                           SVC(kernel='linear'),
                                           DecisionTreeClassifier()],
                              meta_classifier=LogisticRegression())

    model_name_param_dict = {'kNN': (KNeighborsClassifier(),
                                     {'n_neighbors': [5, 25, 55]}),
                             'LR': (LogisticRegression(),
                                    {'C': [0.01, 1, 100]}),
                             'SVM': (SVC(kernel='linear'),
                                     {'C': [0.01, 1, 100]}),
                             'DT': (DecisionTreeClassifier(),
                                    {'max_depth': [50, 100, 150]}),
                             'Stacking': (sclf,
                                          {'kneighborsclassifier__n_neighbors': [5, 25, 55],
                                           'svc__C': [0.01, 1, 100],
                                           'decisiontreeclassifier__max_depth': [50, 100, 150],
                                           'meta-logisticregression__C': [0.01, 1, 100]}),
                             'AdaBoost': (AdaBoostClassifier(),
                                          {'n_estimators': [50, 100, 150, 200]}),
                             'GBDT': (GradientBoostingClassifier(),
                                      {'learning_rate': [0.01, 0.1, 1, 10, 100]}),
                             'RF': (RandomForestClassifier(),
                                    {'n_estimators': [100, 150, 200, 250]})}
