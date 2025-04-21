## yearmade range consolidate 2010 or later
#numbedroms 'Estimate Total 5 or more bedrooms' coosldate 5 and 6
#num rooms conoslidate to 9 and over as max 
#fuel not applicabel is no fuel used

#Importint tools to dealing with missing values and normalizing values
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def vb_transform(RECS_raw):
    """As previously mentioned, the majority of the dataset features were factor variables, many of which needed recoding in order to correctly capture the variation being described.
Nominal factor variables have no intrinsic ordering (dummies).
Ordinal factors have a clear order to their factor levels. Each level (in ranges) denotes a numerically higher value than the previous. We decided that these would be more appropriately formatted in integer/numeric form, which can better convey their continuous range of values, aid interpretation of the variable coefficient, and reduce standard error by decreasing the number of coefficients for model estimation (many algorithms generate a separate model coefficient for each factor level via one-hot-encoding). Cardinal: We left these variables as-is."""

    #divide predictors from y variables
    #resopnse vb
    response_qual = ['SCALEB', 'SCALEG', 'SCALEE', 'ENERGYASST'] #TO DO: others?
    response = response_qual + ['KWH','BTUEL', 'BTUNG', 'BTULP', 'BTUFO', 'TOTALBTU'] #NO COST OF WOOD SO WE DONT TAKE INTO ACCOUTN BTU IN WOOD 'BTUWD'
    
    Y = RECS_raw.loc[:, response] 
    #add all positive instace in  qual varriable, irrespective of frequency of event
    forgoing = {0:0,  1:1, 2:1,  3:1} # 0: Never, 1: Almost every month, 2: Some months, 3: 1 or 2 months
    Y['SCALEB'] = [forgoing[item] for item in Y['SCALEB']] 
    Y['SCALEG'] = [forgoing[item] for item in Y['SCALEG']] 
    Y['SCALEE'] = [forgoing[item] for item in Y['SCALEE']] 
    
    #add other fuels (butane and fuel oil) -no wood!
    Y['BTUOF'] = Y[['BTULP', 'BTUFO']].sum(axis = 1)


    #predictors
    RECS = RECS_raw.drop(columns = response) 
    #changed cells flagged for missing data to nan
    RECS = RECS[RECS.columns].replace({'-2':np.nan, -2:np.nan})
    
    #add other fuels (not NG or EL together)
    RECS['DOLLAROF'] = RECS[['DOLLARLP', 'DOLLARFO']].sum(axis = 1)

    ACS_variables = ['TYPEHUQ', 'IECC_climate_code', 'HDD30YR_PUB', 'CDD30YR_PUB','KOWNRENT', 'YEARMADERANGE', 'BEDROOMS', 'TOTROOMS', 'FUELHEAT',  'NHSLDMEM', 'MONEYPY', 'DIVISION', 'STATE_FIPS', 'UATYP10', 'DOLLAREL', 'DOLLARNG', 'DOLLAROF' , 'NWEIGHT'] #avoid 'TOTALDOL'for collinearity
    RECS_ACS_subset = RECS[ACS_variables]
    
    
    #Small adjustments to consolidate census and RECS variable values
    ##change to owned for people who occupy a residence without paying rent.
    RECS_ACS_subset['KOWNRENT'][RECS_ACS_subset['KOWNRENT']==3] = 1 
    ##numbedrooms consolidate over 5 bedrooms
    RECS_ACS_subset['BEDROOMS'][RECS_ACS_subset['BEDROOMS']==6] = 5 
    ##num members consolidate over 6 people
    RECS_ACS_subset['NHSLDMEM'][RECS_ACS_subset['NHSLDMEM']==7] = 6
    ##rooms consolidate over 9 rooms
    RECS_ACS_subset['TOTROOMS'][RECS_ACS_subset['TOTROOMS'].isin([10,11,12,13,14,15])] = 9 
    ## yearmade range consolidate 2010 or later
    RECS_ACS_subset['YEARMADERANGE'][RECS_ACS_subset['YEARMADERANGE'].isin([8,9])] = 8 
    ## income range consolidate to census groupings
    RECS_ACS_subset['MONEYPY'][RECS_ACS_subset['MONEYPY'].isin([1,2,3])] = 3  #less than 10k
    RECS_ACS_subset['MONEYPY'][RECS_ACS_subset['MONEYPY'].isin([4,5])] = 5 #10-15k
    
    # Define a dictionary to map old values to new values
    map_rent = {1: 'Own', 2: 'Rent'}
    # Use the replace function to change column values
    RECS_ACS_subset['KOWNRENT'] = RECS_ACS_subset['KOWNRENT'].replace(map_rent)
    
    # Define a dictionary to map old values to new values
    map_fuel = {5: 'Electricity', 1: 'Network_gas', 2:'Bottled_gas', 3:'Fuel_oil', 7:'Wood', 99:'Other', -2:'Not applicable'}
    # Use the replace function to change column values
    RECS_ACS_subset['FUELHEAT'] = RECS_ACS_subset['FUELHEAT'].replace(map_fuel)
    
    # Define a dictionary to map old values to new values
    map_htype = {1: 'Mobile_home', 2: 'SF_detached', 3:'SF_attached', 4:'2_to_4_units', 5:'5_or_more_units'}
    # Use the replace function to change column values
    RECS_ACS_subset['TYPEHUQ'] = RECS_ACS_subset['TYPEHUQ'].replace(map_htype)
   
    
    #Ordinal v Cardinal selection
    #We identified which variables should be classified as ordinal or cardinal variables, and created a list containing the variables in each category.
    ordinal = ['YEARMADERANGE', 'MONEYPY', 'NWEIGHT']
    cardinal = ['BEDROOMS', 'TOTROOMS', 'NHSLDMEM', 'CDD30YR_PUB', 'HDD30YR_PUB', 'DOLLAREL', 'DOLLARNG', 'DOLLAROF' ] #avoid 'TOTALDOL'for collinearity

    #rest are nominal
    not_nominal = ordinal + cardinal 
    nominal = RECS_ACS_subset.loc[:, ~RECS_ACS_subset.columns.isin(not_nominal)].copy()

    #Create dummies
    """We one-hot encoded qualitative variables to create dummies, and included drop_first=True to make dummies for one less than the total number of factors for each dummy variable in order to avoid collinearity issues. New factor variables were also created by consolidating information from existing factors into one-hot-encoded binary values"""

    RECS_dummies = pd.get_dummies(data = RECS_ACS_subset, columns= nominal.columns, drop_first=True) 

    #Binning ordinal variables
    """"As mentioned previously, RECS captures several quantitative variables as ordinal variables. For example, YEARMADERANGE represents the year the house was built. Values for this variable include 1 = Before 1950, 2 = 1950-1959, 3 = 1960-1969 and so on. Rather than creating a new dummy variable for each of these factors, we transformed these factors into numeric values. Our reasoning was that numeric values would better represent the fact that the variables are indeed quantitative, and creating a model with a smaller number of features would aid in reducing model variance.
We thus translated the ordinal factors that represented ranges into midpoints (for example, 1950-1959 was translated to 1955). One limitation of this approach is that factors on the high and low end of the scale did not have midpoints, for example, it was unclear which values should correctly estimate the year for "houses built before 1950." For these factors, we made educated guesses. In the large cell below, we translated ordinal variables into numeric values. This factor-to-number recoding was applied to variables including: the year the house was built, household income, when the residents moved into the house, how many stories are in the house, and a large number of variables related to the number of specific household appliances and the age and frequency of use of those appliances."""

    # Define a dictionary to map old values to new values
    map_yr = {1: 1945, 2: 1955, 3: 1965, 4:1975, 5:1985, 6:1995, 7:2005, 8:2015}
    # Use the replace function to change column values
    RECS_dummies['YEARMADERANGE'] = RECS_dummies['YEARMADERANGE'].replace(map_yr)
    
    # Define a dictionary to map old values to new values
    map_income = {3: 5000, 5: 12500, 6:17500, 7:22500, 8:27500, 9:32500, 10:37500, 11:45500, 12:55500, 13:67500, 14:87500, 15:125000,16:175000}
    # Use the replace function to change column values
    RECS_dummies['MONEYPY'] = RECS_dummies['MONEYPY'].replace(map_income)
    


    ####Dealing with NaNs through KNN inputer
    """"In order to minimimize the number of observations we'd need to remove from the dataset, we used KNNImputer to provide a basis for estimating the values of missing features. Indeed, as can be seen,certain columns we were using had substantial numbers of missiong observations (out of 1085)."""
    
    #We applied KNNImputer (with K = 3 nearest neighbors) to estimate values for missing variables, based on the average of the 3 closest samples in the explanatory variables' space.
    imputer = KNNImputer(n_neighbors=3)
    RECS_input = pd.DataFrame(imputer.fit_transform(RECS_dummies),columns = RECS_dummies.columns)

    #normalized independent variables (when regularization methods are applied, features with larger values would not be over-weighted.)
    #scaler = MinMaxScaler()
    #RECS_clean = pd.DataFrame(scaler.fit_transform(RECS_input), columns = RECS_dummies.columns)
    
    #Test code to normalize recs and ACS data
    cols_to_norm = ['HDD30YR_PUB', 'CDD30YR_PUB', 'DOLLAREL', 'DOLLARNG', 'DOLLAROF', 
                                  'YEARMADERANGE', 'BEDROOMS', 'TOTROOMS', 'MONEYPY', 'NHSLDMEM']
    RECS_to_normalize = RECS_input[cols_to_norm]

    # Calculate mean and standard deviation
    mean_values = RECS_to_normalize.mean()
    std_values = RECS_to_normalize.std()
    RECS_norm_param = pd.DataFrame({'Mean': mean_values, 'std': std_values})

    # Normalize the selected columns in the original DataFrame (df)
    RECS_input[cols_to_norm] = (RECS_input[cols_to_norm] - mean_values) / std_values
    
    #dont normalize the weights
    RECS_input['NWEIGHT'] =  RECS_ACS_subset['NWEIGHT'] 

    #Check that there are no NaNs in the dataframe
    #assert RECS_clean.isna().sum().unique()
    
    #delete solar homes from prediction model ? NO
    #RECS_clean = RECS_clean[~RECS_clean.SOLAR == 1]

    return  RECS_input, Y, RECS_norm_param








