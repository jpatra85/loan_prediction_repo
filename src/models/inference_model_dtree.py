# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:12:20 2020

@author: jpatr_000
"""
# =============================================================================
# Red the processed data & train with decision tree to draw inference
# =============================================================================
 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import numpy as np
import pandas as pd
import pickle
import os

df_dt = pd.read_csv("./data/interim/df_preprocessed_out.csv")
df_dt.drop(['Unnamed: 0'] , axis = 1 , inplace = True)

# manually created groups for loan purpose and primary business to be merged 

#df_loan_purpose_groups = pd.read_csv("./data/external/loan_purpose_groups.csv")
#df_primary_busness_groups = pd.read_csv("./data/external/primary_business_groups.csv")
#df_dt = pd.merge(df_dt,df_loan_purpose_groups).drop('loan_count', axis =1)
#df_dt = pd.merge(df_dt,df_primary_busness_groups).drop('loan_count', axis =1)


# selected features after several trails 
df_dt['savings'] = df_dt.annual_income - 12 * df_dt.monthly_expenses

c = ['age', 'sex','house_area',
       'savings',  
       'home_ownership', 'type_of_house', 
       'sanitary_availability', 'water_availabity', 
       'loan_tenure','loan_amount']

# encode columns like sex , water availability  
df_dt = pd.get_dummies(df_dt[c])
df_dt.shape
df_dt.columns


# =============================================================================
# Here we will try out several set samples to find out predictors that are 
# consitantly with greater predictability
# =============================================================================

y = df_dt.pop('loan_amount')
X = df_dt

# keep a validation dataset sperate for final validation
X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.1)


# We iterate over many samples and find the consitant predictors and stabilty 
# of the model 

mape_train = []
mape_test  = []
mape_diff  = []
feat_importances = pd.Series()


for i in range(500):

    X_train, X_test, y_train, y_test = train_test_split(X_trn, y_trn, test_size=0.2)

    # Here we could have varied the the hyoerparameters
    
    clf = DecisionTreeRegressor(max_depth = 15, max_features = 5, 
                                min_samples_leaf = 20)
    clf = clf.fit(X_train, y_train)

    mape_tr = np.mean(abs(y_train - clf.predict(X_train))/y_train)
    mape_ts = np.mean(abs(y_test - clf.predict(X_test))/y_test)
    print("Mape Train", mape_tr)
    print("Mape Test" , mape_ts)
    
    # store the intermediate train and test errors and top 10 predictors for 
    # each iteration
    mape_train.append(mape_tr)
    mape_test.append(mape_ts)
    mape_diff.append(mape_tr - mape_ts)    
    ft = pd.Series(clf.feature_importances_, index=X_train.columns).nlargest(15)
    feat_importances = feat_importances.append(ft)


# we will analyze model stability here 

print(np.mean(mape_train))
print(np.mean(mape_test))

plt.scatter(mape_train,mape_test) # train & test mape closely following eachother
plt.show()
plt.boxplot(mape_train) # not many outlier 
plt.show()
plt.boxplot(mape_test)
plt.show()
plt.hist(mape_diff)     # the difference in train test mape has a small spred
plt.show()


# select the top features the precious iterations

top_features_by_avg_contribution = set(feat_importances.groupby(feat_importances.index).mean().nlargest(5).index)
top_features_by_total_contribution = set(feat_importances.groupby(feat_importances.index).sum().nlargest(5).index)
top_features_by_occurence = set(feat_importances.groupby(feat_importances.index).count().nlargest(5).index)


best_features = list(top_features_by_avg_contribution.union( 
                top_features_by_total_contribution.union(
                top_features_by_occurence)))

# ['water_availabity',
#  'loan_tenure',
#  'sanitary_availability',
#  'house_area',
#  'age',
#  'savings',
#  'home_ownership']


#filename = './models/dtree_model_visualization.sav'
filename = 'notfound.sav'

if os.path.isfile(filename):
    
    # load the model from disk
    clf = pickle.load(open(filename, 'rb'))
    print("Model loaded" , clf)   

else:

    # craete the final model for visualizing the rules the drove loan amount
    # we will compromise with model fit and accuracy in order to keep the tree small
    
    X_train, X_test, y_train, y_test = train_test_split(X_trn[best_features],
                                                        y_trn, test_size=0.2)
    clf = DecisionTreeRegressor(max_depth = 5, max_features = 5,
                                    min_samples_leaf = 700,
                                    min_samples_split = 1500)
    clf = clf.fit(X_train, y_train)

    # save the model to disk
    pickle.dump(clf, open(filename, 'wb'))


# check the accuracy if the out of sample validation shows unstability

mape_tr = np.mean(abs(y_train - clf.predict(X_train))/y_train)
mape_ts = np.mean(abs(y_test - clf.predict(X_test))/y_test)
mape_val = np.mean(abs(y_val - clf.predict(X_val[best_features]))/y_val)
print("Mape Train", mape_tr)
print("Mape Test" , mape_ts)
print("Mape Val" ,  mape_val)

# Mape Train 0.2840719577649216
# Mape Test 0.28513957393071626
# Mape Val 0.28757197906322984

# feature importance
feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns).nlargest(10)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# store dtree visualization as dot file
fn = X_train.columns
export_graphviz(clf,
                 out_file="tree_today.dot",
                 feature_names = fn,
                 filled = True)

