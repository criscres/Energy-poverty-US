import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 
import geopandas as gpd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
import math

from sklearn.model_selection import train_test_split
import timeit

#dependencies from our .py files
import recs_EDA
import regression_recs_sample_weights as reg

warnings.filterwarnings('ignore')


def clean_vbs(RECS_raw):
    
    X_sub, Y, RECS_norm_param = recs_EDA.vb_transform(RECS_raw) 
    
    return X_sub, Y


def prep_eui(RECS_raw, X_sub, surface, e_use):
    
    #construct Y variable
    y_name = "EUI_"+e_use
    
    #Change cases with surface value of 0 to 0 energy use
    RECS_raw[e_use]=np.where(RECS_raw[surface]==0, 0, RECS_raw[e_use])
    RECS_raw[surface]=np.where(RECS_raw[surface]==0, 1, RECS_raw[surface]) #0 dvided by 1 will be 0, to solve indetermination
    
    #calc y
    y=RECS_raw.loc[:,e_use].reset_index(drop = True).div(RECS_raw.loc[:,surface].reset_index(drop = True)) 

    
    return X_sub, y, y_name

def bootstrap_weights_feature_eng(X_sub, y, feature_engineer = True):

    test_prop = 0.2
    boots = False
    if boots == True:
        #save bootstrap proportions
        boot_samples = pd.DataFrame(columns=['boot_num', 'len_Xtrain','prop_18k', 'prop_126M'])

        #drop weights
        #drop weight column
        weights = X_sub.NWEIGHT
        X =  X_sub.drop('NWEIGHT', axis = 1)

        #split with bootstraps
        n = round(math.sqrt(len(X)))
        for b in range(n):

            X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X,y, weights, test_size =test_prop, random_state=b)

            boot_samples.loc[b, 'boot_num'] = b
            boot_samples.loc[b, 'len_Xtrain'] = len(X_train)
            boot_samples.loc[b, 'prop_18k'] = len(X_train)/len(X_sub)
            boot_samples.loc[b, 'prop_126M'] = X_train.NWEIGHT.sum()/X_sub.NWEIGHT.sum()
            #drop weight column
            X = X_sub.drop(columns = ['NWEIGHT'], inplace = True)
    else:
        b = 0
        #drop weight column
        weights = X_sub.NWEIGHT
        X = X_sub.drop('NWEIGHT', axis = 1)
        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X,y, weights, test_size =test_prop, random_state=b)
        
        #weights
        apply_weights = True
        if apply_weights == True:
            pass
        else:
            sw_train = np.ones(len(sw_train))
            sw_test = np.ones(len(sw_test))  
    
    #Feature Engineering 
    n = len(X.columns)

    if feature_engineer == True:
        interactions = 1+0.5*n+0.5*n**2 - 1 #intercept

        poly = PolynomialFeatures(interaction_only=True,include_bias = False)
        X_train_eng = poly.fit_transform(X_train)
        X_test_eng = poly.fit_transform(X_test)

        X_train = pd.DataFrame(X_train_eng, columns = poly.get_feature_names_out(X_train.columns))
        X_test = pd.DataFrame(X_test_eng, columns = poly.get_feature_names_out(X_test.columns))
    else:
        interactions = n

   
    return X_train, X_test, y_train, y_test, sw_train, sw_test 

def run_models(summary_table):
    
    #Linear regressions
    Models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'] # list of models
    m_sub = dict.fromkeys(Models) #save models here for prediction


    for Model, i in zip(Models, np.arange(len(Models))):
        summary_table, m_sub[Models[i]]  = reg.regularization_fit(Model, X_train, y_train, X_test, y_test, sw_train, sw_test, summary_table, efficiency)

    #KNN with CV
    splits = 5#20
    knn_cv, sqrt_sub_mses_knn, mses_sub_knn, r2_sub_knn, summary_table= reg.KNN_fit(X_train, y_train, X_test, y_test, sw_train, sw_test, summary_table, splits)
    
    #ANN
    clf, score_sub, summary_table = reg.ANN_fit(X_train, y_train, X_test, y_test, sw_train, sw_test, summary_table)
    
    #Ensemble methods 
    bag, rf, gb, ab, summary_table = reg.ensemble_fit(X_train, y_train,  X_test, y_test, sw_train, sw_test, summary_table)
    
    return summary_table


if __name__ == "__main__":
    
    #import RECS
    path = '/global/scratch/users/cristina_crespo/p1_data/'
    RECS_raw = pd.read_csv(path +'RECS/recs2020_public_v5.csv')
    
    #select dependent variable
    efficiency = True
    y_name = "TOTALBTU"#   (does not include wood)
    #TOTALBTU BTUEL, BTUNG, BTUOF
    
    #Set up summary dataframe to compare models
    summary_table = pd.DataFrame(columns = ['Model','R2', 'MSE', 'RMSE','MAE', 'MAPE']) 
    
    
    #if efficiency caluclated, prepare dep variable
    if y_name == "TOTALBTU" and efficiency == True:
        X_sub, Y = clean_vbs(RECS_raw)
        #clean recs
        surface = 'TOTCSQFT' #TOTHSQFT, TOTCSQFT
        e_use = 'BTUELCOL' #TOTALBTUSPH, BTUELCOL #Calibrated electricity usage for space cooling (central air conditioning, individual units, and evaporative coolers), in thousand Btu, 2020 (dont include BTUELAHUCOL #Calibrated electricity usage furnace fans used for cooling  in thousand Btu, 2020
        X_sub, y, y_name = prep_eui(RECS_raw, X_sub,surface, e_use )

    else: 
        #clean recs
        X_sub, Y = clean_vbs(RECS_raw)
        y=Y[y_name].reset_index(drop = True) 
    
    #bootstrap weigths, and feature engineering
    feature_engineer = False
    X_train, X_test, y_train, y_test, sw_train, sw_test = bootstrap_weights_feature_eng(X_sub, y, feature_engineer)
    
    #run models
    summary_table_final = run_models(summary_table)

    #save model comparisons
    summary_table_final.to_csv(path+'out_final/model_comparison/'+y_name+'_fe=0.csv') 
    
    

