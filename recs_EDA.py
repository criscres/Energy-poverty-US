

#Importint tools to dealing with missing values and normalizing values
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def recs_filtering(RECS_raw, filter_climate = False):

	"""In addition, in this step we began to subset the data to include our predictor variables only. As such, we dropped features including:
	- those that do not provide us with meaningful information for making predictions -- for example, the household's unique identifier, the kWh to Btu conversion factor, etc. 
	- y variables
	- climate regions (which we have already used to subset the data)
	- we also dropped race during this phase, as we decided that it would be valuable to use this variable after making predictions in order to explore predicted racial disparities in energy poverty (our y variable). Although we did not have time to do this during this analysis, this would be one potential area for further analysis.
	- we changed cells that were flagged as having missing data to nan, with plans to synthetically impute these variables later on. """

	if filter:
		#subset CA CLIMATE_REGION_PUB, IECC_CLIMATE_PUB, or REGIONC
		RECS_region = RECS_raw[RECS_raw['DIVISION']==10]
	else:
		RECS_region = RECS_raw

	#drop variables that do not provide useful predictive information
	RECS = RECS_region.drop(columns = ['DOEID','NWEIGHT', 'ELXBTU', 'NGXBTU','FOXBTU', 'LPXBTU', 'PERIODEL', 'PERIODNG', 'PERIODFO', 'PERIODLP']) #Final weight for respresntation, conversion kwh btu, period of data from energy supplier provided (ordinal)
	#dropped race
	RECS = RECS.drop(columns = 'HOUSEHOLDER_RACE')
	#We also drop the climate regions since we already subsetted accordingly
	RECS = RECS.drop(columns = ['CLIMATE_REGION_PUB', 'IECC_CLIMATE_PUB', 'DIVISION', 'REGIONC'])

	#Identified qualitative y variables
	response_qual = ['SCALEB', 'SCALEG', 'SCALEE', 'NOHEATDAYS', 'NOACDAYS']
	#RECS_all = RECS_all.drop(columns = response_qual)
	#cardinal: 'NOHEATDAYS', NOACDAYS

	response = response_qual + ['KWH','BTUEL', 'BTUNG','WOODBTU', 'PELLETBTU', 'TOTALBTU'] #Identified quantitative y variables
	RECS = RECS.drop(columns = response) 
	Y = RECS_region.loc[:, response] #Dropped quantitative and qualitative y variables

	#Dropped y variables that are subdivisions of the main y variable (energy consumption), such as electricity or gas usage associated with a specific appliance.
	RECS = RECS.loc[:, ~RECS.columns.str.startswith('Z')]
	RECS = RECS.loc[:, ~RECS.columns.str.startswith('BTU')]
	RECS = RECS.loc[:, ~RECS.columns.str.startswith('DOL')]
	RECS = RECS.loc[:, ~RECS.columns.str.startswith('KWH')]
	RECS = RECS.loc[:, ~RECS.columns.str.startswith('BRR')]
	RECS = RECS.loc[:, ~RECS.columns.str.startswith('GALLON')]
	RECS = RECS.loc[:, ~RECS.columns.str.startswith('CUFEE')]
	RECS = RECS.loc[:, ~RECS.columns.str.startswith('TOTALBTU')]

	#changed cells flagged for missing data to nan
	RECS = RECS[RECS.columns].replace({'-2':np.nan, -2:np.nan})

	#Identify missingness
	max_miss = 0.90
	miss_variables =  RECS.columns[RECS.isna().mean() > max_miss].to_list()
	check_miss = miss_variables+['TOTALBTU'] 

	#Drop variables (columns) that have over 90% missingness
	#drop missingness in variables
	RECS = RECS.drop(columns = miss_variables)
	np.shape(RECS)

	return RECS, RECS_region, check_miss, Y

def vb_transform(RECS, Y):

	"""As previously mentioned, the majority of the dataset features were factor variables, many of which needed recoding in order to correctly capture the variation being described.
	Nominal factor variables have no intrinsic ordering (dummies).
	Ordinal factors have a clear order to their factor levels. Each level (in ranges) denotes a numerically higher value than the previous. We decided that these would be more appropriately formatted in integer/numeric form, which can better convey their continuous range of values, aid interpretation of the variable coefficient, and reduce standard error by decreasing the number of coefficients for model estimation (many algorithms generate a separate model coefficient for each factor level via one-hot-encoding).
	Cardinal: We left these variables as-is."""


	#Ordinal v Cardinal selection

	#We identified which variables should be classified as ordinal or cardinal variables, and created a list containing the variables in each category.
	ordinal = ['YEARMADERANGE', 'OCCUPYYRANGE', 'WINDOWS', 'AGERFRI1', 
	           'AGERFRI2','AGEFRZR', 'NUMMEAL', 'AGEDW', 'AGECWASH', 
	           'AGECDRYER', 'TVSIZE1', 'TVONWD1', 'TVONWE1', 'TVSIZE2', 
	           'TVONWD2', 'TVONWE2', 'EQUIPAGE', 'AGECENAC', 'WWACAGE',
	          'WHEATAGE','LGTINNUM','LGTINCAN', 'LGTINCFL', 'LGTINLED',
	          'LGTOUTNUM', 'MONEYPY', 'STORIES', 'SIZRFRI1', 'SIZRFRI2', 
	          'ENERGYASST', 'ENERGYASST11','ENERGYASST12','ENERGYASST13','ENERGYASST14','ENERGYASST15'] #these are categorical but already one-hot encoded

	cardinal = ['TYPEGLASS', 'CABLESAT', 'COMBODVR', 'SEPDVR', 'PLAYSTA', 'DVD', 'VCR',
	           'INTSTREAM', 'TVAUDIOSYS', 'DESKTOP', 'NUMLAPTOP', 'NUMTABLET', 'ELPERIPH', 'NUMSMPHONE', 'CELLPHONE',
	           'TEMPHOME', 'TEMPGONE', 'TEMPNITE', 'USEMOISTURE', 'NUMBERAC', 'TEMPHOMEAC', 'TEMPGONEAC', 'TEMPNITEAC',
	            'NUMCFAN', 'NUMFLOORFAN', 'NUMWHOLEFAN', 'NUMATTICFAN','LGTIN4', 'HHAGE', 'NHSLDMEM', 'NUMADULT',
	           'NUMCHILD', 'ATHOME', 'TOTCSQFT', 'TOTHSQFT','TOTSQFT_EN', 'TOTUCSQFT','TOTUSQFT', 'CDD30YR','CDD65', 'CDD80',
	           'TOTALDOL','TOTALDOLSPH','TOTALDOLWTH','TOTALDOLCOK','TOTALDOLCDR','TOTALDOLPL','TOTALDOLHTB','TOTALDOLNEC',
	           'HDD30YR','HDD65','HDD50','GNDHDD65','WSF','OA_LAT','GWT','DBT1','DBT99', 'WOODAMT','PELLETAMT', 'BEDROOMS', 'TOTROOMS'] #typeglass included caus emore insaltuon natural order

	#rest are nominal
	not_nominal = ordinal + cardinal 
	nominal = RECS.loc[:, ~RECS.columns.isin(not_nominal)].copy()


	#Create dummies
	"""We one-hot encoded qualitative variables to create dummies, and included drop_first=True to make dummies for one less than the total number of factors for each dummy variable in order 
	to avoid collinearity issues. New factor variables were also created by consolidating information from existing factors into one-hot-encoded binary values such as presence of heated pool, 
	use of electricity for space heating, etc."""

	RECS['KOWNRENT'][RECS['KOWNRENT']==3] = 1 #change to owned for people who occupy a residence without paying rent.

	RECS_dummies = pd.get_dummies(data = RECS, columns= nominal.columns, drop_first=True) 

	RECS_dummies.head()

	#Binning ordinal variables

	""""As mentioned previously, RECS captures several quantitative variables as ordinal variables. For example, YEARMADERANGE represents the year the house was built. Values for this variable include 1 = Before 1950, 2 = 1950-1959, 3 = 1960-1969 and so on. Rather than creating a new dummy variable for each of these factors, we transformed these factors into numeric values. Our reasoning was that numeric values would better represent the fact that the variables are indeed quantitative, and creating a model with a smaller number of features would aid in reducing model variance.
	We thus translated the ordinal factors that represented ranges into midpoints (for example, 1950-1959 was translated to 1955). One limitation of this approach is that factors on the high and low end of the scale did not have midpoints, for example, it was unclear which values should correctly estimate the year for "houses built before 1950." For these factors, we made educated guesses.
	In the large cell below, we translated ordinal variables into numeric values. This factor-to-number recoding was applied to variables including: the year the house was built, household income, when the residents moved into the house, how many stories are in the house, and a large number of variables related to the number of specific household appliances and the age and frequency of use of those appliances."""

	YEARMADERANGE_values = [1925, 1955, 1965, 1975, 1985, 1995, 2005, 2013]
	RECS_dummies['YEARMADERANGE'] = [YEARMADERANGE_values[i-1] for i in np.array(RECS_dummies['YEARMADERANGE'].astype(np.int))]

	OCCUPYYRANGE_values = [1945, 1955, 1965, 1975, 1985, 1995, 2005, 2013]
	RECS_dummies['OCCUPYYRANGE'] = [OCCUPYYRANGE_values[i-1] for i in np.array(RECS_dummies['OCCUPYYRANGE'].astype(np.int))]

	WINDOWS_values = {10: 1.5, 20: 4, 30: 7.5, 41: 13, 42: 17.5, 50: 25, 60: 40}
	RECS_dummies['WINDOWS'] = [WINDOWS_values[i] for i in np.array(RECS_dummies['WINDOWS'].astype(np.int))]

	#inserted NaN for 'not applicable' entries
	AGERFRI1_values = {1: 1, 2: 3, 3: 7, 41: 12, 42: 17, 5: 23, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['AGERFRI1'].notna(), 'AGERFRI1']  = [AGERFRI1_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['AGERFRI1'].notna(), 'AGERFRI1']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	AGERFRI2_values = {1: 1, 2: 3, 3: 7, 41: 12, 42: 17, 5: 23, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['AGERFRI2'].notna(), 'AGERFRI2']  = [AGERFRI2_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['AGERFRI2'].notna(), 'AGERFRI2']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	AGEFRZR_values = {1: 1, 2: 3, 3: 7, 41: 12, 42: 17, 5: 23, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['AGEFRZR'].notna(), 'AGEFRZR']  = [AGEFRZR_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['AGEFRZR'].notna(), 'AGEFRZR']).astype(np.int)]

	#Values are in number of meals cooked per week (ie, 3x/day = 21x/wk)
	NUMMEAL_values = {1: 21, 2: 14, 3: 7, 4: 3.5, 5: 1, 6: .5, 0: 0}
	RECS_dummies['NUMMEAL'] = [NUMMEAL_values[i] for i in np.array(RECS_dummies['NUMMEAL'].astype(np.int))]

	#inserted NaN for 'not applicable' entries
	AGEDW_values = {1: 1, 2: 3, 3: 7, 41: 12, 42: 17, 5: 23, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['AGEDW'].notna(), 'AGEDW']  = [AGEDW_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['AGEDW'].notna(), 'AGEDW']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	AGECWASH_values = {1: 1, 2: 3, 3: 7, 41: 12, 42: 17, 5: 23, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['AGECWASH'].notna(), 'AGECWASH']  = [AGECWASH_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['AGECWASH'].notna(), 'AGECWASH']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	AGECDRYER_values = {1: 1, 2: 3, 3: 7, 41: 12, 42: 17, 5: 23, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['AGECDRYER'].notna(), 'AGECDRYER']  = [AGECDRYER_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['AGECDRYER'].notna(), 'AGECDRYER']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	TVSIZE1_values = {1: 25, 2: 33.5, 3: 49.5, 4: 65, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['TVSIZE1'].notna(), 'TVSIZE1']  = [TVSIZE1_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['TVSIZE1'].notna(), 'TVSIZE1']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	TVONWD1_values = {1: 0.5, 2: 2, 3: 5, 4: 8.5, 5: 12, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['TVONWD1'].notna(), 'TVONWD1']  = [TVONWD1_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['TVONWD1'].notna(), 'TVONWD1']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	TVONWE1_values = {1: 0.5, 2: 2, 3: 5, 4: 8.5, 5: 12, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['TVONWE1'].notna(), 'TVONWE1']  = [TVONWE1_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['TVONWE1'].notna(), 'TVONWE1']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	TVSIZE2_values = {1: 25, 2: 33.5, 3: 49.5, 4: 65, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['TVSIZE2'].notna(), 'TVSIZE2']  = [TVSIZE2_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['TVSIZE2'].notna(), 'TVSIZE2']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	TVONWD2_values = {1: 0.5, 2: 2, 3: 5, 4: 8.5, 5: 12, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['TVONWD2'].notna(), 'TVONWD2']  = [TVONWD2_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['TVONWD2'].notna(), 'TVONWD2']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	TVONWE2_values = {1: 0.5, 2: 2, 3: 5, 4: 8.5, 5: 12, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['TVONWE2'].notna(), 'TVONWE2']  = [TVONWE2_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['TVONWE2'].notna(), 'TVONWE2']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	EQUIPAGE_values = {1: 1, 2: 3, 3: 7, 41: 12, 42: 17, 5: 23, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['EQUIPAGE'].notna(), 'EQUIPAGE']  = [EQUIPAGE_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['EQUIPAGE'].notna(), 'EQUIPAGE']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	AGECENAC_values = {1: 1, 2: 3, 3: 7, 41: 12, 42: 17, 5: 23, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['AGECENAC'].notna(), 'AGECENAC']  = [AGECENAC_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['AGECENAC'].notna(), 'AGECENAC']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	WWACAGE_values = {1: 1, 2: 3, 3: 7, 41: 12, 42: 17, 5: 23, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['WWACAGE'].notna(), 'WWACAGE']  = [WWACAGE_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['WWACAGE'].notna(), 'WWACAGE']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	WHEATAGE_values = {1: 1, 2: 3, 3: 7, 41: 12, 42: 17, 5: 23, -2: np.nan}
	RECS_dummies['WHEATAGE'] = [WHEATAGE_values[i] for i in np.array(RECS_dummies['WHEATAGE'].astype(np.int))]

	LGTINNUM_values = [10, 30, 50, 70, 90]
	RECS_dummies['LGTINNUM'] = [LGTINNUM_values[i-1] for i in np.array(RECS_dummies['LGTINNUM'].astype(np.int))]

	LGTINCAN_values = {1: 100, 2: 75, 3: 50, 4: 25, 0: 0}
	RECS_dummies['LGTINCAN'] = [LGTINCAN_values[i] for i in np.array(RECS_dummies['LGTINCAN'].astype(np.int))]

	LGTINCFL_values = {1: 100, 2: 75, 3: 50, 4: 25, 0: 0}
	RECS_dummies['LGTINCFL'] = [LGTINCFL_values[i] for i in np.array(RECS_dummies['LGTINCFL'].astype(np.int))]

	LGTINLED_values = {1: 100, 2: 75, 3: 50, 4: 25, 0: 0}
	RECS_dummies['LGTINLED'] = [LGTINLED_values[i] for i in np.array(RECS_dummies['LGTINLED'].astype(np.int))]

	#inserted NaN for 'not applicable' entries
	LGTOUTNUM_values = {0:0, 1: 2.5, 2:7, 3:15, 4: np.nan}
	RECS_dummies.loc[RECS_dummies['LGTOUTNUM'].notna(), 'LGTOUTNUM']  = [LGTOUTNUM_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['LGTOUTNUM'].notna(), 'LGTOUTNUM']).astype(np.int)]

	MONEYPY_values = [10000, 30000, 50000, 70000, 90000, 110000, 130000, 250000]
	RECS_dummies['MONEYPY'] = [MONEYPY_values[i-1] for i in np.array(RECS_dummies['MONEYPY'].astype(np.int))]

	#inserted NaN for 'not applicable' entries; assumed split-level home is three levels.
	STORIES_values = {10: 1, 20: 2, 31: 3, 32: 4, 40: 3, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['STORIES'].notna(), 'STORIES']  = [STORIES_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['STORIES'].notna(), 'STORIES']).astype(np.int)]

	#Assumed mini fridge is 3 cubic feet
	#inserted NaN for 'not applicable' entries
	SIZRFRI1_values = {1: 3, 2: 10, 3: 20, 4: 26, 5: 32, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['SIZRFRI1'].notna(), 'SIZRFRI1']  = [SIZRFRI1_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['SIZRFRI1'].notna(), 'SIZRFRI1']).astype(np.int)]

	#inserted NaN for 'not applicable' entries
	SIZRFRI2_values = {1: 3, 2: 10, 3: 20, 4: 26, 5: 32, -2: np.nan}
	RECS_dummies.loc[RECS_dummies['SIZRFRI2'].notna(), 'SIZRFRI2']  = [SIZRFRI2_values[i] for i in np.array(RECS_dummies.loc[RECS_dummies['SIZRFRI2'].notna(), 'SIZRFRI2']).astype(np.int)]


	####Dealing with NaNs through KNN inputer

	""""In order to minimimize the number of observations we'd need to remove from the dataset, we used KNNImputer to provide a basis for estimating the values of missing features. Indeed, as can be seen, 
	certain columns we were using had substantial numbers of missiong observations (out of 1085)."""

	RECS_dummies.loc[:, RECS_dummies.columns.isin(not_nominal)].isna().sum()

	#We applied KNNImputer (with K = 3 nearest neighbors) to estimate values for missing variables, based on the average of the 3 closest samples in the explanatory variables' space.
	imputer = KNNImputer(n_neighbors=3)
	RECS_input = pd.DataFrame(imputer.fit_transform(RECS_dummies),columns = RECS_dummies.columns)

	#normalized independent variables (when regularization methods are applied, features with larger values would not be over-weighted.)
	scaler = MinMaxScaler()
	RECS_clean = pd.DataFrame(scaler.fit_transform(RECS_input), columns = RECS_dummies.columns)
	Y_norm = Y#pd.DataFrame(scaler.fit_transform(Y), columns = Y.columns)

	# Check that there are no NaNs in the dataframe
	#assert RECS_clean.isna().sum().unique()

	return  RECS_clean, Y_norm, cardinal #norm is not normalized here

def vb_ACS_subset(RECS_clean, ACS_variables):
	#Selecting variables in RECS that are present in ACS
	RECS_ACS_subset = RECS_clean[ACS_variables]

	return  RECS_ACS_subset


