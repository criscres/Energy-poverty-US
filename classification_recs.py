
import numpy as np
import pandas as pd

#KNN
from sklearn.neighbors import KNeighborsRegressor

#Ensemble methods
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

#SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV  

def ensemble_fit(X_all_train, y_all_train, X_all_val, y_all_val, X_sub_train, y_sub_train, X_sub_val, y_sub_val, y_all_test, X_all_test, y_sub_test, X_sub_test):
	
	rand_seed = 5

	#Bagging (no hyperparameters)BaggingClassifier(

	#All
	bag_tree_all = BaggingClassifier(random_state=rand_seed, n_estimators = 500, max_samples = 500, max_features = 500)
	bag_tree_all.fit(X_all_train, y_all_train)

	bag_train_score_all = bag_tree_all.score(X_all_train, y_all_train)
	bag_val_score_all = bag_tree_all.score(X_all_val, y_all_val)

	print("Bagging")
	print('[All] Train Score: ', bag_train_score_all)
	print('[All] Validation Score: ', bag_val_score_all)

	#Sub
	bag_tree_sub = BaggingClassifier(random_state=rand_seed, n_estimators = 500, max_samples = 500, max_features = 10)
	bag_tree_sub.fit(X_sub_train, y_sub_train)

	bag_train_score_sub = bag_tree_sub.score(X_sub_train, y_sub_train)
	bag_val_score_sub = bag_tree_sub.score(X_sub_val, y_sub_val)

	print('[Sub] Train Score: ', bag_train_score_sub)
	print('[Sub] Validation Score: ', bag_val_score_sub)

	#Random Forest (no hyperparameters)

	#All
	rf_tree_all = RandomForestClassifier(random_state=rand_seed, max_depth=15, n_estimators = 600, max_samples = 500, max_features = 500,min_samples_leaf=4)
	rf_tree_all.fit(X_all_train, y_all_train)

	rf_train_score_all = rf_tree_all.score(X_all_train, y_all_train)
	rf_val_score_all = rf_tree_all.score(X_all_val, y_all_val)

	print("Random Forest")
	print('[All] Train Score: ', rf_train_score_all)
	print('[All] Validation Score: ', rf_val_score_all)

	#Sub
	rf_tree_sub = RandomForestClassifier(random_state=rand_seed, max_depth=15, n_estimators = 600, max_samples = 500, max_features = 10,min_samples_leaf=4)
	rf_tree_sub.fit(X_sub_train, y_sub_train)

	rf_train_score_sub = rf_tree_sub.score(X_sub_train, y_sub_train)
	rf_val_score_sub = rf_tree_sub.score(X_sub_val, y_sub_val)

	print('[Sub] Train Score: ', rf_train_score_sub)
	print('[Sub] Validation Score: ', rf_val_score_sub)

	#Boosting

	#All
	gb_tree_all =  GradientBoostingClassifier(random_state=rand_seed)

	param_dist = {'n_estimators': randint(2, 20),
	              'max_depth': randint(1, 10)}

	rnd_gb_search = RandomizedSearchCV(gb_tree_all, param_distributions=param_dist, 
	                                cv=5, n_iter=200, random_state = 2020)

	rnd_gb_search.fit(X_all_train, y_all_train)

	print("Boosting")
	print(rnd_gb_search.best_params_)

	gb_tree_all =  GradientBoostingClassifier(random_state=2020, n_estimators=rnd_gb_search.best_params_['n_estimators'], max_depth=rnd_gb_search.best_params_['max_depth'])
	gb_tree_all.fit(X_all_train, y_all_train)

	gb_train_score_all = gb_tree_all.score(X_all_train, y_all_train)
	gb_val_score_all = gb_tree_all.score(X_all_val, y_all_val)

	print('[All] Train Score: ', gb_train_score_all)
	print('[All] Validation Score: ', gb_val_score_all)

	#Sub
	gb_tree_sub =  GradientBoostingClassifier(random_state=rand_seed)

	param_dist = {'n_estimators': randint(2, 20),
	              'max_depth': randint(1, 10)}

	rnd_gb_search = RandomizedSearchCV(gb_tree_sub, param_distributions=param_dist, 
	                                cv=5, n_iter=200, random_state = 2020)

	rnd_gb_search.fit(X_sub_train, y_sub_train)

	print(rnd_gb_search.best_params_)

	gb_tree_sub =  GradientBoostingClassifier(random_state=2020, n_estimators=rnd_gb_search.best_params_['n_estimators'], max_depth=rnd_gb_search.best_params_['max_depth'])
	gb_tree_sub.fit(X_sub_train, y_sub_train)

	gb_train_score_sub = gb_tree_sub.score(X_sub_train, y_sub_train)
	gb_val_score_sub = gb_tree_sub.score(X_sub_val, y_sub_val)

	print('[Sub] Train Score: ', gb_train_score_sub)
	print('[Sub] Validation Score: ', gb_val_score_sub)


	return bag_train_score_all, bag_val_score_all, bag_train_score_sub, bag_val_score_sub, rf_train_score_all, rf_val_score_all, rf_train_score_sub, rf_val_score_sub, gb_train_score_all, gb_val_score_all, gb_train_score_sub, gb_val_score_sub, bag_tree_all, rf_tree_all, gb_tree_all, bag_tree_sub, rf_tree_sub, gb_tree_sub

def SVC_fit(X_sub_train,y_sub_train, X_sub_test, y_sub_test,X_all_train,y_all_train,X_all_test,y_all_test):
	
	param_grid = {'C': [10, 15, 20], 'kernel': ['rbf', 'poly', 'sigmoid'], 'gamma': [1]}
	grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
	grid.fit(X_sub_train,y_sub_train)
	print('Best SVC parameters: ', grid.best_estimator_)

	y_sub_pred = grid.predict(X_sub_test)
	print("Sub")
	print(confusion_matrix(y_sub_test,y_sub_pred))
	print(classification_report(y_sub_test,y_sub_pred))

	param_grid = {'C': [10, 15, 20], 'kernel': ['rbf', 'poly', 'sigmoid'], 'gamma': [1]}
	grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
	grid.fit(X_all_train,y_all_train)
	print('Best SVC parameters: ', grid.best_estimator_)

	y_all_pred = grid.predict(X_all_test)
	print("All")
	print(confusion_matrix(y_all_test,y_all_pred))
	print(classification_report(y_all_test,y_all_pred))

	return


	