# -*- coding: utf-8 -*-

"""

    文件名:    main.py
    功能：     主程序

    手机价格预测
    任务：建立不同的集成学习模型进行手机价格等级预测，并使用交叉验证选择最优的超参数

    数据集来源： https://www.kaggle.com/vikramb/mobile-price-eda-prediction

    
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV


import config


def inspect_dataset(train_data, test_data):
    """
        查看数据集
    """
    print('\n===================== 数据查看 =====================')
    print('训练集有{}条记录。'.format(len(train_data)))
    print('测试集有{}条记录。'.format(len(test_data)))

    # 可视化各类别的数量统计图
    plt.figure(figsize=(10, 5))

    # 训练集
    ax1 = plt.subplot(1, 2, 1)
    sns.countplot(x='price_range', data=train_data)

    plt.title('Training Data')
    plt.xticks(rotation='vertical')
    plt.xlabel('Price Range')
    plt.ylabel('Count')

    # 测试集
    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x='price_range', data=test_data)

    plt.title('Test Data')
    plt.xticks(rotation='vertical')
    plt.xlabel('Price Range')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def do_feature_engineering(train_data, test_data):
    """
        特征工程：处理训练集和测试集中的特征
        1. 类别特征，使用独热编码（One-Hot Encoding）
        2. 数值特征，使用标准化（Standardization）或归一化（Normalization）
    """
    # 类别特征
    trn_cat_feats = train_data[config.cat_cols].values
    tes_cat_feats = test_data[config.cat_cols].values

    # 数值特征
    trn_numeric_feats = train_data[config.numeric_cols]
    tes_numeric_feats = test_data[config.numeric_cols]

    # 独热编码
    enc = OneHotEncoder(sparse=False)
    enc_trn_cat_feats = enc.fit_transform(trn_cat_feats)
    enc_tes_cat_feats = enc.transform(tes_cat_feats)

    trn_all_feats = np.hstack((trn_numeric_feats, enc_trn_cat_feats))
    tes_all_feats = np.hstack((tes_numeric_feats, enc_tes_cat_feats))

    # 标准化
    std_scaler = StandardScaler()
    scaled_trn_all_feats = std_scaler.fit_transform(trn_all_feats)
    scaled_tes_all_feats = std_scaler.transform(tes_all_feats)

    return scaled_trn_all_feats, scaled_tes_all_feats


def train_model(X_train, y_train, X_test, y_test, model_name, model, param_range):
    """

        根据给定的参数训练模型，并返回
        1. 最优模型
        2. 平均训练耗时
        3. 准确率
    """
    print('训练{}...'.format(model_name))
    clf = GridSearchCV(estimator=model,
                       param_grid=param_range,
                       cv=5,
                       scoring='accuracy',
                       refit=True)
    start = time.time()
    clf.fit(X_train, y_train)
    # 计时
    end = time.time()
    duration = end - start
    print('耗时{:.4f}s'.format(duration))

    # 验证模型
    print('训练准确率：{:.3f}'.format(clf.score(X_train, y_train)))

    score = clf.score(X_test, y_test)
    print('测试准确率：{:.3f}'.format(score))
    print('训练模型耗时: {:.4f}s'.format(duration))
    print()

    return clf, score, duration


def main():
    """
        主函数
    """
    # 加载数据
    all_data = pd.read_csv(os.path.join(config.dataset_path, 'data.csv'))
    train_data, test_data = train_test_split(all_data, test_size=1/3, random_state=10)

    # 数据查看
    inspect_dataset(train_data, test_data)

    # 构建训练测试数据
    # 特征处理
    X_train, X_test = do_feature_engineering(train_data, test_data)

    print('共有{}维特征。'.format(X_train.shape[1]))

    # 标签处理
    y_train = train_data[config.label_col].values
    y_test = test_data[config.label_col].values

    # 数据建模及验证
    print('\n===================== 数据建模及验证 =====================')

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

    # 比较结果的DataFrame
    results_df = pd.DataFrame(columns=['Accuracy (%)', 'Time (s)'],
                              index=list(model_name_param_dict.keys()))
    results_df.index.name = 'Model'

    for model_name, (model, param_range) in model_name_param_dict.items():
        best_clf, best_acc, mean_duration = train_model(X_train, y_train, X_test, y_test,
                                                        model_name, model, param_range)
        results_df.loc[model_name, 'Accuracy (%)'] = best_acc * 100
        results_df.loc[model_name, 'Time (s)'] = mean_duration

    results_df.to_csv(os.path.join(config.output_path, 'model_comparison.csv'))

    # 模型及结果比较
    print('\n===================== 模型及结果比较 =====================')

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    results_df.plot(y=['Accuracy (%)'], kind='bar', ylim=[50, 100], ax=ax1, title='Accuracy(%)', legend=False)

    ax2 = plt.subplot(1, 2, 2)
    results_df.plot(y=['Time (s)'], kind='bar', ax=ax2, title='Time (s)', legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_path, './pred_results.png'))
    plt.show()


if __name__ == '__main__':
    main()
