
import numpy as np
import pandas as pd

import regression_recs as reg
import classification_recs as classif

from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

from sklearn.ensemble import RandomForestClassifier

def regression_prediction(RECS_ACS_subset, y, pred_name, acs_formatted, acs_data_raw):
	test_prop = 0.2
	rand_seed = 5

	#_btu for TOTALBTU _all for all X variables, _sub for subset X variables also in ACS
	X_sub_train, X_sub_test, y_sub_train, y_sub_test = reg.get_X_y(RECS_ACS_subset, y, test_prop, rand_seed) # get X and y train and test dataframes for full dataset

	#Lasso
	alphas_lasso = np.linspace(1, 1000, 100)
	kf = KFold(n_splits = 5, shuffle = True, random_state = 9)
	model = LassoCV(cv = kf, alphas=alphas_lasso)
	        
	model.fit(X_sub_train, y_sub_train)

	#regularization_fit(Model, X_train, X_test, y_train, y_test):

	predicted_acs_lasso=model.predict(acs_formatted)
	acs_data_raw[pred_name]=predicted_acs_lasso

	return acs_data_raw

def classification_prediction(RECS_clean, RECS_ACS_subset, y_qual, pred_name, acs_formatted, acs_data_raw):
	test_prop = 0.2
	rand_seed = 5

	X_all, X_all_test, y_all, y_all_test, X_all_train, X_all_val, y_all_train, y_all_val, X_sub, X_sub_test, y_sub, y_sub_test, X_sub_train, X_sub_val, y_sub_train, y_sub_val = reg.get_X_y_val(RECS_clean, RECS_ACS_subset, y_qual, test_prop, rand_seed)

	#Sub
	rf_tree_sub = RandomForestClassifier(random_state=rand_seed, max_depth=15, n_estimators = 600, max_samples = 500, max_features = 10,min_samples_leaf=4)
	rf_tree_sub.fit(X_sub_train, y_sub_train)

	rf_train_score_sub = rf_tree_sub.score(X_sub_train, y_sub_train)
	rf_val_score_sub = rf_tree_sub.score(X_sub_val, y_sub_val)


	print('Train Score: ', rf_train_score_sub)
	print('Validation Score: ', rf_val_score_sub)
	acs_y=rf_tree_sub.predict(acs_formatted)
	acs_data_raw[pred_name]=acs_y

	return acs_data_raw

def indicator_calc(acs_data_raw):
	##### 1. High share of energy expenditure in income (2M)

	acs_data_raw["Energy Burden"] = acs_data_raw["Avg. Annual Energy Cost"]  / acs_data_raw["Aggregate Household Income in the Past 12 Months"] 

	median = acs_data_raw["Energy Burden"].median()

	#boolean
	acs_data_raw["boolean_2M"] = acs_data_raw["Energy Burden"]>2*median
	#level
	acs_data_raw["2M"] = (acs_data_raw["Energy Burden"]-2*median)/2*median

	##### 2. Low absolute energy expenditure (M/2): 

	#boolean
	acs_data_raw["boolean_M/2"] = acs_data_raw["Energy Burden"]<(median/2)
	#level
	acs_data_raw["M_2"] = (acs_data_raw["Energy Burden"]-(median/2))/(median/2)

	##### 3. Inability to keep home adequately warm or having to forego necessities

	#This is already calculated and saved in acs_data_raw as `SCALEB_predicted`

	##### 4. Estimated Residential Energy Consumption in kBtu

	#This is already calculated and saved in acs_data_raw as `TOTALBTU_predicted`
	return acs_data_raw