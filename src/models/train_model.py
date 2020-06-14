# =============================================================================
# Read the features in and split into train test 
# =============================================================================

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import os
from loan_prediction_repo.src.models import model_linear_L2 as mdl



df_final = pd.read_csv("./data/processed/df_ft_out.csv")
df_final.drop(['Unnamed: 0'] , axis = 1 , inplace = True)

y = df_final.pop('loan_amount')
X = df_final

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# =============================================================================
# Train a random forest model
# =============================================================================

filename = './models/rf_model.sav'

if os.path.isfile(filename):
    
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X_test, y_test)
    print("result : ", result)
    
else:
    
    rf = RandomForestRegressor(random_state = 42)
    
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': [90],
        'max_features': [5 , 7],
        'min_samples_leaf': [3],
        'min_samples_split': [8],
        'n_estimators': [300 ]
    }
    
    # Create a model
    
    rf = RandomForestRegressor()# Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = 3, n_jobs = -1, verbose = 2)
    
    grid_search.fit(X_train, y_train)
    best_grid = grid_search.best_estimator_
    print(grid_search.best_params_)
    
    # {'bootstrap': True,
    #  'max_depth': 90,
    #  'max_features': 3,
    #  'min_samples_leaf': 3,
    #  'min_samples_split': 8,
    #  'n_estimators': 300}
    
    # save the model to disk
    filename = './models/rf_model.sav'
    pickle.dump(best_grid, open(filename, 'wb'))
    loaded_model = best_grid

# evaluate the model
p_train = loaded_model.predict(X_train)
mape_train = np.mean(abs(1 - p_train/y_train))
print("train mape :", mape_train)                # 21.3%

p_test = loaded_model.predict(X_test)
mape_test = np.mean(abs(p_test - y_test)/y_test) 
print("test mape : ", mape_test)                      #22.5 % 
print("accuracy ", 100 * (1 -mape_test))
plt.scatter(p_test, y_test)

# feature importance
feat_importances = pd.Series(loaded_model.feature_importances_, index=df_final.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()
feat_importances.nsmallest(10).plot(kind='barh')
plt.show()


# df_final["pred"] = loaded_model.predict(df_final)
# df_final["actual"] = y
# df_final.to_csv("c:/Users/jpatr_000/loan_pred_fnal_out.csv")


# =============================================================================
# Linear ression model with L2 regularization 
# =============================================================================

# normalize the data

mean_train_X = np.mean(X_test)
sd_train_X   = np.std(X_test)
mean_y       = np.mean(y_train)
sd_y         = np.std(y_train)   

X_train_z = (X_train - mean_train_X) / sd_train_X
y_train_z = (y_train - mean_y)/sd_y  
X_test_z  = (X_test  - mean_train_X) / sd_train_X


# invoke the model
linL2 = mdl.Model_Linear_L2()
model = linL2.gradient_descentt(X_train_z, y_train_z ,iteration= 1000, lumbda = 0.04 , learning_rate = 0.001) 


#check accuracy
pred_z = np.dot(X_test_z,model['param'])
pred = (pred_z + mean_y) *  sd_y
mape = np.mean(abs(y_test - pred)/y_test)
print(mape)
plt.scatter(pred, y_test)