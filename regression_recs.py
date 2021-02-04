
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing regression operations
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

#Ensemble methods
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

#KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score


#Create splits
def get_X_y(X, y, test_prop, rand_seed):
    """
    This function returns four dataframes containing the testing and training X and y values used in land-use regression.
    Input: df, a Pandas dataframe with all of the fields in the land-use regression dataset; 
        cols_to_drop, a list of the names (strings) of the columns to drop from df in order to obtain the feature variables.
        y_col, a column name (as a string) of df that represents the response variable
        test_prop, a float between 0 and 1 indicating the fraction of the data to include in the test split
        rand_seed, an integer, used to define the random state
    Returns: X_train, X_test, y_train, y_test, four dataframes containing the training and testing subsets of the 
    feature matrix X and response matrix y
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =test_prop, random_state=rand_seed)
    
    return X_train, X_test, y_train, y_test

def get_X_y_val(RECS_clean, RECS_ACS_subset, y, test_prop, rand_seed):
    
    #All
    # make the test/train split
	X_all, X_all_test, y_all, y_all_test = train_test_split(RECS_clean,y, test_size = 0.20, random_state=rand_seed)
	# make the train/validation split
	X_all_train, X_all_val, y_all_train, y_all_val = train_test_split(X_all,y_all, test_size = 0.20, random_state=rand_seed)

	#Subset
	# make the test/train split
	X_sub, X_sub_test, y_sub, y_sub_test = train_test_split(RECS_ACS_subset,y, test_size = 0.20, random_state=rand_seed)
	# make the train/validation split
	X_sub_train, X_sub_val, y_sub_train, y_sub_val = train_test_split(X_sub,y_sub, test_size = 0.20, random_state=rand_seed)
	
	return X_all, X_all_test, y_all, y_all_test, X_all_train, X_all_val, y_all_train, y_all_val, X_sub, X_sub_test, y_sub, y_sub_test, X_sub_train, X_sub_val, y_sub_train, y_sub_val


def regularization_fit(Model, X_train, X_test, y_train, y_test, save=False):

    """
    This function fits a model of type Model to the data in the training set of X and y, and finds the MSE on the test set
    Inputs: 
        Model (sklearn model): the type of sklearn model with which to fit the data - LinearRegression, Ridge, or Lasso
        X_train: the set of features used to train the model
        y_train: the set of response variable observations used to train the model
        X_test: the set of features used to test the model
        y_test: the set of response variable observations used to test the model
        alpha: the penalty parameter, to be used with Ridge and Lasso models only
    """   
    
    if Model == LinearRegression:
        model = Model()
        
    elif Model == Ridge:
        alphas_ridge = np.linspace(0.0001,10, 100)
        kf = KFold(n_splits = 5, shuffle = True, random_state = 9)
        model = RidgeCV(cv = kf, alphas=alphas_ridge)
        #alpha_opt_ridge = model.alpha_
        
    else: #Lasso
        alphas_lasso = np.linspace(1, 1000, 100)
        kf = KFold(n_splits = 5, shuffle = True, random_state = 9)
        model = LassoCV(cv = kf, alphas=alphas_lasso)
        #alpha_opt_lasso = model.alpha_
        
    model.fit(X_train, y_train)
    #if save:
    #predicted_acs_lasso=model.predict(acs_formatted)

    mse = mean_squared_error(y_test, model.predict(X_test))
    coef = model.coef_.flatten()
    r2 = r2_score(y_test, model.predict(X_test))
    
    
    if Model == LinearRegression:
        pass
    else:
        alpha_opt = model.alpha_
        print("optimal alpha "+str(Model)+":", alpha_opt)
    
    return mse, coef, r2

#Ensemble methods

def importance_plot(rf_tree, bo_tree, X):

    #For bagging tree.feature_importances_ does not exist, we need totake tha average of all the trees manually
    #feature_importance_bag = pd.DataFrame({'Feature': X_sub.columns, 'Importance': tree.feature_importances_})# get the importance of each feature
    
    #feature_importance_bag = np.mean([tree.feature_importances_ for tree in bag_tree.estimators_], axis=0)
    
    feature_importance_rf = pd.DataFrame({'Feature': X.columns, 'Importance': rf_tree.feature_importances_})# get the importance of each feature
    feature_importance_bo = pd.DataFrame({'Feature': X.columns, 'Importance': bo_tree.feature_importances_})# get the importance of each feature
    
    #calculate the relative feature importance  & sort
    #relative_importance = feature_importance_bag["Importance"] / max(feature_importance_bag["Importance"])*100
    #feat_df_bag = pd.DataFrame({'Feature': X_sub.columns, 'Relative Importance': relative_importance})
    #feat_df_bag = feat_df_bag.sort_values(by="Relative Importance", ascending=True)
    
    relative_importance = feature_importance_rf["Importance"] / max(feature_importance_rf["Importance"])*100
    feat_df_rf = pd.DataFrame({'Feature': X.columns, 'Relative Importance': relative_importance})
    feat_df_rf = feat_df_rf.sort_values(by="Relative Importance", ascending=True)
    
    relative_importance = feature_importance_bo["Importance"] / max(feature_importance_bo["Importance"])*100
    feat_df_bo = pd.DataFrame({'Feature': X.columns, 'Relative Importance': relative_importance})
    feat_df_bo = feat_df_bo.sort_values(by="Relative Importance", ascending=True)
    
    fig, axes = plt.subplots(1, 2, sharex=True, figsize =(5,5))

    fig.add_subplot(111, frameon=False)
    fig.subplots_adjust( wspace=2, hspace=None)
    #axes[0].barh(y=feat_df_bag["Feature"],width=feat_df_bag["Relative Importance"])
    axes[0].barh(y=feat_df_rf["Feature"],width=feat_df_rf["Relative Importance"])
    axes[1].barh(y=feat_df_bo["Feature"],width=feat_df_bo["Relative Importance"])
    
    plt.xlabel('Relative feature importance', fontsize=10)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.show()

def ensemble_fit(X_all_train, y_all_train, X_all_val, y_all_val, X_sub_train, y_sub_train, X_sub_val, y_sub_val, y_all_test, X_all_test, y_sub_test, X_sub_test):
	
	rand_seed = 5

	#Bagging (no hyperparameters)BaggingClassifier(

	#All
	bag_tree_all = BaggingRegressor(random_state=rand_seed, n_estimators = 300, max_samples = 500, max_features = 500)
	bag_tree_all.fit(X_all_train, y_all_train)

	bag_train_score_all = bag_tree_all.score(X_all_train, y_all_train)
	bag_val_score_all = bag_tree_all.score(X_all_val, y_all_val)

	print("Bagging")
	print('[All] Train Score: ', bag_train_score_all)
	print('[All] Validation Score: ', bag_val_score_all)

	#Sub
	bag_tree_sub = BaggingRegressor(random_state=rand_seed, n_estimators = 300, max_samples = 400, max_features = 10)
	bag_tree_sub.fit(X_sub_train, y_sub_train)

	bag_train_score_sub = bag_tree_sub.score(X_sub_train, y_sub_train)
	bag_val_score_sub = bag_tree_sub.score(X_sub_val, y_sub_val)

	print('[Sub] Train Score: ', bag_train_score_sub)
	print('[Sub] Validation Score: ', bag_val_score_sub)

	r2_all_bag = r2_score(y_all_test, bag_tree_all.predict(X_all_test))
	r2_sub_bag = r2_score(y_sub_test, bag_tree_sub.predict(X_sub_test))
	print('r2_all_bag: ', r2_all_bag)
	print('r2_sub_bag: ', r2_sub_bag)

	#Random Forest (no hyperparameters)

	#All
	rf_tree_all = RandomForestRegressor(random_state=rand_seed, max_depth=15, n_estimators = 600, max_samples = 500, max_features = 500,min_samples_leaf=4)
	rf_tree_all.fit(X_all_train, y_all_train)

	rf_train_score_all = rf_tree_all.score(X_all_train, y_all_train)
	rf_val_score_all = rf_tree_all.score(X_all_val, y_all_val)

	print("Random Forest")
	print('[All] Train Score: ', rf_train_score_all)
	print('[All] Validation Score: ', rf_val_score_all)

	#Sub
	rf_tree_sub = RandomForestRegressor(random_state=rand_seed, max_depth=15, n_estimators = 600, max_samples = 500, max_features = 10,min_samples_leaf=4)
	rf_tree_sub.fit(X_sub_train, y_sub_train)

	rf_train_score_sub = rf_tree_sub.score(X_sub_train, y_sub_train)
	rf_val_score_sub = rf_tree_sub.score(X_sub_val, y_sub_val)

	print('[Sub] Train Score: ', rf_train_score_sub)
	print('[Sub] Validation Score: ', rf_val_score_sub)

	r2_all_rf = r2_score(y_all_test, rf_tree_all.predict(X_all_test))
	r2_sub_rf = r2_score(y_sub_test, rf_tree_sub.predict(X_sub_test))
	print('r2_all_rf: ', r2_all_rf)
	print('r2_sub_rf: ', r2_sub_rf)

	#Boosting

	#All
	gb_tree_all =  GradientBoostingRegressor(random_state=rand_seed)

	param_dist = {'n_estimators': randint(2, 20),
	              'max_depth': randint(1, 10)}

	rnd_gb_search = RandomizedSearchCV(gb_tree_all, param_distributions=param_dist, 
	                                cv=5, n_iter=200, random_state = 2020)

	rnd_gb_search.fit(X_all_train, y_all_train)

	print("Boosting")
	print(rnd_gb_search.best_params_)

	gb_tree_all =  GradientBoostingRegressor(random_state=2020, n_estimators=rnd_gb_search.best_params_['n_estimators'], max_depth=rnd_gb_search.best_params_['max_depth'])
	gb_tree_all.fit(X_all_train, y_all_train)

	gb_train_score_all = gb_tree_all.score(X_all_train, y_all_train)
	gb_val_score_all = gb_tree_all.score(X_all_val, y_all_val)

	print('[All] Train Score: ', gb_train_score_all)
	print('[All] Validation Score: ', gb_val_score_all)

	#Sub
	gb_tree_sub =  GradientBoostingRegressor(random_state=rand_seed)

	param_dist = {'n_estimators': randint(2, 20),
	              'max_depth': randint(1, 10)}

	rnd_gb_search = RandomizedSearchCV(gb_tree_sub, param_distributions=param_dist, 
	                                cv=5, n_iter=200, random_state = 2020)

	rnd_gb_search.fit(X_sub_train, y_sub_train)

	print(rnd_gb_search.best_params_)

	gb_tree_sub =  GradientBoostingRegressor(random_state=2020, n_estimators=rnd_gb_search.best_params_['n_estimators'], max_depth=rnd_gb_search.best_params_['max_depth'])
	gb_tree_sub.fit(X_sub_train, y_sub_train)

	gb_train_score_sub = gb_tree_sub.score(X_sub_train, y_sub_train)
	gb_val_score_sub = gb_tree_sub.score(X_sub_val, y_sub_val)

	print('[Sub] Train Score: ', gb_train_score_sub)
	print('[Sub] Validation Score: ', gb_val_score_sub)

	r2_all_gb = r2_score(y_all_test, gb_tree_all.predict(X_all_test))
	r2_sub_gb = r2_score(y_sub_test, gb_tree_sub.predict(X_sub_test))
	print('r2_all_gb: ', r2_all_gb)
	print('r2_sub_gb: ', r2_sub_gb)

	return r2_all_bag, r2_sub_bag, bag_train_score_all, bag_val_score_all, bag_train_score_sub, bag_val_score_sub, r2_all_rf, r2_sub_rf, rf_train_score_all, rf_val_score_all, rf_train_score_sub, rf_val_score_sub, r2_all_gb, r2_sub_gb, gb_train_score_all, gb_val_score_all, gb_train_score_sub, gb_val_score_sub, bag_tree_all, rf_tree_all, gb_tree_all, bag_tree_sub, rf_tree_sub, gb_tree_sub
	
def KNN_fit(X_sub_train, y_sub_train, X_all_train, y_all_train, X_sub_test, y_sub_test, X_all_test, y_all_test):

	mses_sub_knn = []
	mses_all_knn = []
	for n in range(0,20):
	    knn_cv = KNeighborsRegressor(n_neighbors=n+1)
	    kfold = KFold(n_splits=10)
	    #kfold = KFold(n_splits=10, random_state = 7)
	    scoring = 'neg_mean_squared_error'
	    resultArr_sub = cross_val_score(knn_cv, X_sub_train, y_sub_train, cv=kfold, scoring=scoring)
	    resultArr_all = cross_val_score(knn_cv, X_all_train, y_all_train, cv=kfold, scoring=scoring)
	    mse_sub = resultArr_sub.mean()
	    mse_all = resultArr_all.mean()
	    mses_sub_knn.append(mse_sub)
	    mses_all_knn.append(mse_all)

	sqrt_sub_mses_knn = np.sqrt(np.abs(mses_sub_knn))
	sqrt_all_mses_knn = np.sqrt(np.abs(mses_all_knn))

	K_sub = mses_sub_knn.index(max(mses_sub_knn)) +1
	K_all = mses_all_knn.index(max(mses_all_knn)) +1


	knn_cv = KNeighborsRegressor(n_neighbors=K_sub)
	knn_cv.fit(X_sub_train, y_sub_train)

	r2_sub_knn = r2_score(y_sub_test, knn_cv.predict(X_sub_test))
	print('r2_sub_knn: ', r2_sub_knn)


	knn_cv = KNeighborsRegressor(n_neighbors=K_all)
	knn_cv.fit(X_all_train, y_all_train)

	r2_all_knn = r2_score(y_all_test, knn_cv.predict(X_all_test))
	print('r2_all_knn: ', r2_all_knn)

	return sqrt_sub_mses_knn, sqrt_all_mses_knn, mses_sub_knn, mses_all_knn, r2_all_knn, r2_sub_knn

	
	
	
