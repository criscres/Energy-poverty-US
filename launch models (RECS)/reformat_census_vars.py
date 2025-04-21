
import numpy as np
import pandas as pd

def avg_income(df_summary):
    #weighted mean income 
    df_cols = df_summary.rename(columns={'Estimate Total Less than $10,000':'Less than \$10,000',
                                         'Estimate Total $10,000 to $14,999':'\$10,000 to \$14,999',
                                         'Estimate Total $15,000 to $19,999':'\$15,000 to \$19,999',
                                         'Estimate Total $20,000 to $24,999':'\$20,000 to \$24,999',
                                         'Estimate Total $25,000 to $29,999':'\$25,000 to \$29,999',
                                         'Estimate Total $30,000 to $34,999':'\$30,000 to \$34,999',
                                         'Estimate Total $35,000 to $39,999':'\$35,000 to \$39,999',
                                         'Estimate Total $40,000 to $44,999':'\$40,000 to \$44,999',
                                         'Estimate Total $45,000 to $49,999':'\$45,000 to \$49,999',
                                         'Estimate Total $50,000 to $59,999':'\$50,000 to \$59,999',
                                         'Estimate Total $60,000 to $74,999':'\$60,000 to \$74,999',
                                         'Estimate Total $75,000 to $99,999':'\$75,000 to \$99,999',
                                         'Estimate Total $100,000 to $124,999':'\$100,000 to \$124,999',
                                         'Estimate Total $125,000 to $149,999':'\$125,000 to \$149,999',
                                         'Estimate Total $150,000 to $199,999':'\$150,000 to \$199,999',
                                         'Estimate Total $200,000 or more':'\$200,000 or more'})
    
    #reformat for RECS options
    df_cols['\\$40,000 to \\$49,999'] = df_cols['\\$40,000 to \\$44,999'] + df_cols['\\$45,000 to \\$49,999'] 
    df_cols['\\$100,000 to \\$149,999'] = df_cols['\\$100,000 to \\$124,999'] + df_cols['\\$125,000 to \\$149,999'] 
    df_cols['\\$150,000 or more'] = df_cols['\\$150,000 to \\$199,999'] + df_cols['\\$200,000 or more'] 
    
    #subset
    df_cols = df_cols[['Less than \\$10,000',
                     '\\$10,000 to \\$14,999',
                     '\\$15,000 to \\$19,999',
                     '\\$20,000 to \\$24,999',
                     '\\$25,000 to \\$29,999',
                     '\\$30,000 to \\$34,999',
                     '\\$35,000 to \\$39,999',
                     '\\$40,000 to \\$49,999',
                     '\\$50,000 to \\$59,999',
                     '\\$60,000 to \\$74,999',
                     '\\$75,000 to \\$99,999',
                     '\\$100,000 to \\$149,999',
                     '\\$150,000 or more']]

    #normalize by total number of HHs
    df_cols = df_cols.div(df_cols.sum(axis=1), axis=0)
    #replacement options, in order    
    new_vals = [5000, 12500, 17500, 22500, 27500, 32500, 37500, 45500, 55500, 67500, 87500, 125000,175000]
    #apply weights
    for i in range(len(df_cols.columns)):
        df_cols.iloc[:,i] = df_cols.iloc[:,i] * new_vals[i]

    #calculate weighted average
    df_cols["Average household income"] = df_cols.sum(axis=1)
    #merge on index qith original dataframe
    df_summary = df_summary.merge(df_cols["Average household income"], how='outer', left_index=True, right_index=True)

    #rename median values 
    df_summary.rename(columns = {'Estimate Median household income in the past 12 months (in 2020 inflation-adjusted dollars)': 'Median household income'}, inplace = True)

    #Filter out Average household income of 0 or median income cannot be calculated
    df_summary = df_summary[(df_summary[ 'Average household income']!=0) & 
                         (df_summary['Median household income']!=-666666666)]
    return  df_summary

def avg_hh_members(df_summary):
    
   #weighted HH members
    df_cols = df_summary.rename(columns={'Estimate Total Family households 2-person household':'Family: 2-person',
                                         'Estimate Total Family households 3-person household':'Family: 3-person',
                                         'Estimate Total Family households 4-person household':'Family: 4-person',
                                         'Estimate Total Family households 5-person household':'Family: 5-person',
                                         'Estimate Total Family households 6-person household':'Family: 6-person',
                                         'Estimate Total Family households 7-or-more person household':'Family: 7-or-more-person',
                                         'Estimate Total Nonfamily households 1-person household':'Nonfamily: 1-person',
                                         'Estimate Total Nonfamily households 2-person household':'Nonfamily: 2-person',
                                         'Estimate Total Nonfamily households 3-person household':'Nonfamily: 3-person',
                                         'Estimate Total Nonfamily households 4-person household':'Nonfamily: 4-person',
                                         'Estimate Total Nonfamily households 5-person household':'Nonfamily: 5-person',
                                         'Estimate Total Nonfamily households 6-person household':'Nonfamily: 6-person',
                                         'Estimate Total Nonfamily households 7-or-more person household':'Nonfamily: 7-or-more-person'})
    #subset
    df_cols = df_cols[['Family: 2-person','Family: 3-person', 'Family: 4-person','Family: 5-person', 'Family: 6-person', 'Family: 7-or-more-person',
                    'Nonfamily: 1-person', 'Nonfamily: 2-person', 'Nonfamily: 3-person', 'Nonfamily: 4-person', 'Nonfamily: 5-person',
                      'Nonfamily: 6-person', 'Nonfamily: 7-or-more-person']]


    #normalize by total number of HHs
    df_cols = df_cols.astype('int')
    df_cols = df_cols.div(df_cols.sum(axis=1), axis=0)
    #replacement options, in order    
    new_vals = [2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7]
    #apply weights
    for i in range(len(df_cols.columns)):
        df_cols.iloc[:,i] = df_cols.iloc[:,i] * new_vals[i]

    #calculate weighted average
    df_cols['Average household members'] = df_cols.sum(axis=1)
    #merge on index qith original dataframe
    df_summary  = df_summary.merge(df_cols['Average household members'], how='outer', left_index=True, right_index=True)

    return  df_summary


def dep_ratios(df_summary):
    
    #Dep ratios
    df_cols = df_summary[['Estimate Total Male','Estimate Total Male Under 5 years','Estimate Total Male 5 to 9 years','Estimate Total Male 10 to 14 years',
     'Estimate Total Male 15 to 17 years','Estimate Total Male 18 and 19 years','Estimate Total Male 20 years','Estimate Total Male 21 years','Estimate Total Male 22 to 24 years',
     'Estimate Total Male 25 to 29 years','Estimate Total Male 30 to 34 years','Estimate Total Male 35 to 39 years','Estimate Total Male 40 to 44 years','Estimate Total Male 45 to 49 years',
     'Estimate Total Male 50 to 54 years','Estimate Total Male 55 to 59 years','Estimate Total Male 60 and 61 years','Estimate Total Male 62 to 64 years','Estimate Total Male 65 and 66 years','Estimate Total Male 67 to 69 years',
     'Estimate Total Male 70 to 74 years','Estimate Total Male 75 to 79 years','Estimate Total Male 80 to 84 years','Estimate Total Male 85 years and over','Estimate Total Female',
     'Estimate Total Female Under 5 years','Estimate Total Female 5 to 9 years','Estimate Total Female 10 to 14 years',
     'Estimate Total Female 15 to 17 years','Estimate Total Female 18 and 19 years','Estimate Total Female 20 years','Estimate Total Female 21 years','Estimate Total Female 22 to 24 years',
     'Estimate Total Female 25 to 29 years','Estimate Total Female 30 to 34 years','Estimate Total Female 35 to 39 years',
     'Estimate Total Female 40 to 44 years','Estimate Total Female 45 to 49 years','Estimate Total Female 50 to 54 years','Estimate Total Female 55 to 59 years',
     'Estimate Total Female 60 and 61 years','Estimate Total Female 62 to 64 years','Estimate Total Female 65 and 66 years','Estimate Total Female 67 to 69 years',
     'Estimate Total Female 70 to 74 years','Estimate Total Female 75 to 79 years','Estimate Total Female 80 to 84 years','Estimate Total Female 85 years and over',]]


    df_cols = df_cols.astype('int')

    #Children pop
    df_cols['Total children'] = df_cols['Estimate Total Male Under 5 years'] + df_cols['Estimate Total Male 5 to 9 years'] +  df_cols['Estimate Total Male 10 to 14 years']
    + df_cols['Estimate Total Female Under 5 years'] + df_cols['Estimate Total Female 5 to 9 years'] +  df_cols['Estimate Total Female 10 to 14 years']
    #Old age pop
    df_cols['Total old age'] = df_cols['Estimate Total Male 65 and 66 years'] +  df_cols['Estimate Total Male 67 to 69 years'] + df_cols['Estimate Total Male 70 to 74 years'] +  df_cols['Estimate Total Male 75 to 79 years'] +  df_cols['Estimate Total Male 80 to 84 years'] +  df_cols['Estimate Total Male 85 years and over']
    + df_cols['Estimate Total Female 65 and 66 years'] +  df_cols['Estimate Total Female 67 to 69 years'] + df_cols['Estimate Total Female 70 to 74 years'] +  df_cols['Estimate Total Female 75 to 79 years'] +  df_cols['Estimate Total Female 80 to 84 years'] +  df_cols['Estimate Total Female 85 years and over']                           

    #Denominator of dependency ratio (population 15-64)
    df_cols['Total 15-64'] = df_cols['Estimate Total Male'] + df_cols['Estimate Total Female']- df_cols['Total children'] #-  df_cols['Total old age']

    #Child dependency ratio
    df_cols['Child dependency ratio'] = df_cols['Total children']/ df_cols['Total 15-64']

    #Old age dependency ratio
    df_cols['Old age dependency ratio'] = df_cols['Total old age']/ df_cols['Total 15-64']


    #merge on index qith original dataframe
    df_summary  = df_summary.merge(df_cols[['Total children', 'Total 15-64', 'Total old age', 'Child dependency ratio', 'Old age dependency ratio']], how='outer', left_index=True, right_index=True)
    return  df_summary


def hh_gender(df_summary):
    #HH gender 
    df_cols = df_summary[['Estimate Total In households Householder Male',
                          'Estimate Total In households Householder Female']]


    df_cols = df_cols.astype('int')

    #Denominator of dependency ratio (population 15-64)
    df_cols['Householder_binary'] = np.where(df_cols['Estimate Total In households Householder Male'] > df_cols['Estimate Total In households Householder Female'] , 'Male', 'Female') #if = considered female houshold
    df_cols['Householder_female_level'] = df_cols['Estimate Total In households Householder Female'].div(df_cols['Estimate Total In households Householder Male'] + df_cols['Estimate Total In households Householder Female'])


    #merge on index qith original dataframe
    df_summary  = df_summary.merge(df_cols[['Householder_binary', 'Householder_female_level']], how='outer', left_index=True, right_index=True)

    
    return  df_summary

def unit_str(df_summary):
    
    #Units in structure 
    df_cols = df_summary[['Estimate Total 1, detached','Estimate Total 1, attached',
    'Estimate Total 2','Estimate Total 3 or 4','Estimate Total 5 to 9',
    'Estimate Total 10 to 19','Estimate Total 20 to 49',
    'Estimate Total 50 or more', 'Estimate Total Mobile home']]

    df_cols = df_cols.astype('int')

    #reformat for RECS options
    df_cols['2 to 4 units'] = df_cols['Estimate Total 2'] + df_cols['Estimate Total 3 or 4'] 
    df_cols['5 or more units'] = df_cols['Estimate Total 5 to 9'] + df_cols['Estimate Total 10 to 19'] + df_cols['Estimate Total 20 to 49'] + df_cols['Estimate Total 50 or more'] 
    df_cols = df_cols.rename(columns = {'Estimate Total Mobile home': 'Mobile home', 'Estimate Total 1, detached': 'Detached, 1 unit', 'Estimate Total 1, attached': 'Attached, 1 unit'})
    df_cols = df_cols.drop(['Estimate Total 2', 'Estimate Total 3 or 4', 'Estimate Total 5 to 9', 'Estimate Total 10 to 19', 'Estimate Total 20 to 49','Estimate Total 50 or more'], axis=1)

    #normalize by total number of HHs
    df_cols = df_cols.div(df_cols.sum(axis=1), axis=0)


    #merge on index qith original dataframe
    df_summary  = df_summary.merge(df_cols[['Mobile home', 'Detached, 1 unit', 'Attached, 1 unit', '2 to 4 units', '5 or more units']], how='outer', left_index=True, right_index=True)

    return  df_summary

def tenure_type(df_summary):
    #Tenure type
    df_cols = df_summary.rename(columns={'Estimate Total Owner occupied':'Owner occupied',
                                         'Estimate Total Renter occupied': 'Renter occupied'})
                                         
        
    df_cols = df_cols[['Owner occupied',
     'Renter occupied']]

    df_cols = df_cols.astype('int')

    #normalize by total number of HHs
    df_cols = df_cols.div(df_cols.sum(axis=1), axis=0)


    #merge on index qith original dataframe
    df_summary  = df_summary.merge(df_cols[['Owner occupied','Renter occupied']], how='outer', left_index=True, right_index=True)

    return  df_summary

def race_ethn(df_summary):
    #Race & ethnicity 
    df_cols = df_summary.rename(columns={'Estimate Total Not Hispanic or Latino White alone':'Not Hispanic or Latino: White',
     'Estimate Total Not Hispanic or Latino Black or African American alone': 'Not Hispanic or Latino: Black or African American',
     'Estimate Total Not Hispanic or Latino American Indian and Alaska Native alone':'Not Hispanic or Latino: American Indian and Alaska Native',
     'Estimate Total Not Hispanic or Latino Asian alone': 'Not Hispanic or Latino: Asian alone',
     'Estimate Total Not Hispanic or Latino Native Hawaiian and Other Pacific Islander alone':  'Not Hispanic or Latino: Native Hawaiian and Other Pacific Islander',
     'Estimate Total Not Hispanic or Latino Some other race alone': 'Not Hispanic or Latino: Some other race alone',
     'Estimate Total Not Hispanic or Latino Two or more races': 'Not Hispanic or Latino: Two or more races',
      'Estimate Total Hispanic or Latino':'Hispanic or Latino'})


    df_cols = df_cols[['Not Hispanic or Latino: White', 'Not Hispanic or Latino: Black or African American', 
                        'Not Hispanic or Latino: American Indian and Alaska Native', 'Not Hispanic or Latino: Asian alone', 
                        'Not Hispanic or Latino: Native Hawaiian and Other Pacific Islander','Not Hispanic or Latino: Some other race alone', 
                        'Not Hispanic or Latino: Two or more races','Hispanic or Latino']]

    df_cols = df_cols.astype('int')

    #normalize by total number of HHs
    df_cols = df_cols.div(df_cols.sum(axis=1), axis=0)


    #merge on index qith original dataframe
    df_summary  = df_summary.merge(df_cols[['Not Hispanic or Latino: White', 'Not Hispanic or Latino: Black or African American', 
                        'Not Hispanic or Latino: American Indian and Alaska Native', 'Not Hispanic or Latino: Asian alone', 
                        'Not Hispanic or Latino: Native Hawaiian and Other Pacific Islander','Not Hispanic or Latino: Some other race alone', 
                        'Not Hispanic or Latino: Two or more races','Hispanic or Latino']], how='outer', left_index=True, right_index=True)

    return  df_summary

def yr_built(df_summary):
    #Year built
    df_cols = df_summary.rename(columns={'Estimate Total Built 2014 or later': 'Built 2014 or later',
     'Estimate Total Built 2010 to 2013': 'Built 2010-2013',
     'Estimate Total Built 2000 to 2009': 'Built 2000-2009',
     'Estimate Total Built 1990 to 1999': 'Built 1990-1999',
     'Estimate Total Built 1980 to 1989': 'Built 1980-1989',
     'Estimate Total Built 1970 to 1979': 'Built 1970-1979',
     'Estimate Total Built 1960 to 1969': 'Built 1960-1969',
     'Estimate Total Built 1950 to 1959': 'Built 1950-1959',
     'Estimate Total Built 1940 to 1949': 'Built 1940-1949',
     'Estimate Total Built 1939 or earlier': 'Built 1939 or earlier',
     'Estimate Median year structure built': 'Median year built'})


    #reformat like recs
    df_cols['Before 1950'] = df_cols['Built 1939 or earlier'] + df_cols['Built 1940-1949'] 
    df_cols['1950 to 1959'] = df_cols['Built 1950-1959'] 
    df_cols['1960 to 1969'] = df_cols['Built 1960-1969'] 
    df_cols['1970 to 1979'] = df_cols['Built 1970-1979'] 
    df_cols['1980 to 1989'] = df_cols['Built 1980-1989'] 
    df_cols['1990 to 1999'] = df_cols['Built 1990-1999']
    df_cols['2000 to 2009'] = df_cols['Built 2000-2009']
    df_cols['2010 or later'] = df_cols['Built 2010-2013'] + df_cols['Built 2014 or later'] 

    #subset
    df_cols = df_cols[['2010 or later', '2000 to 2009', '1990 to 1999', '1980 to 1989','1970 to 1979',
     '1960 to 1969', '1950 to 1959', 'Before 1950']]

    df_cols = df_cols.astype('int')

    #normalize by total number of HHs
    df_cols = df_cols.div(df_cols.sum(axis=1), axis=0)

    df_midpoints_lst = [2015, 2005, 1995, 1985, 1975, 1965, 1955, 1945]

    for i in range(len(df_cols.columns)):
        df_cols.iloc[:,i] = df_cols.iloc[:,i] * df_midpoints_lst[i]

    #calculate weighted average
    df_cols["Average year built"] = df_cols.sum(axis=1)

    #merge on index with original dataframe
    df_summary  = df_summary.merge(df_cols['Average year built'], how='outer', left_index=True, right_index=True)

    return  df_summary

def num_bedrooms(df_summary):
    #Num Bedrooms
    df_cols = df_summary.rename(columns={'Estimate Total No bedroom': '0 bedrooms',
                               'Estimate Total 1 bedroom': '1 bedroom',
                               'Estimate Total 2 bedrooms': '2 bedrooms',
                               'Estimate Total 3 bedrooms': '3 bedrooms',
                               'Estimate Total 4 bedrooms': '4 bedrooms',
                               'Estimate Total 5 or more bedrooms': '5 or more bedrooms'})

    #subset
    df_cols = df_cols[['0 bedrooms','1 bedroom', '2 bedrooms','3 bedrooms', '4 bedrooms','5 or more bedrooms']]

    df_cols = df_cols.astype('int')

    #normalize by total number of HHs
    df_cols = df_cols.div(df_cols.sum(axis=1), axis=0)

    df_midpoints_lst = [0, 1, 2, 3, 4, 5]

    for i in range(len(df_cols.columns)):
        df_cols.iloc[:,i] = df_cols.iloc[:,i] * df_midpoints_lst[i]

    #calculate weighted average
    df_cols["Average num bedrooms"] = df_cols.sum(axis=1)

    #merge on index with original dataframe
    df_summary  = df_summary.merge(df_cols['Average num bedrooms'], how='outer', left_index=True, right_index=True)
    return  df_summary

def num_rooms(df_summary):
    #Num Rooms
    df_cols = df_summary.rename(columns={'Estimate Total 1 room': '1 room',
                                            'Estimate Total 2 rooms': '2 rooms',
                                            'Estimate Total 3 rooms': '3 rooms',
                                            'Estimate Total 4 rooms': '4 rooms',
                                            'Estimate Total 5 rooms': '5 rooms',
                                            'Estimate Total 6 rooms': '6 rooms',
                                            'Estimate Total 7 rooms': '7 rooms',
                                            'Estimate Total 8 rooms': '8 rooms',
                                            'Estimate Total 9 or more rooms': '9 or more rooms'})

    #subset
    df_cols = df_cols[['1 room', '2 rooms','3 rooms', '4 rooms','5 rooms', '6 rooms','7 rooms', '8 rooms', '9 or more rooms']]

    df_cols = df_cols.astype('int')

    #normalize by total number of HHs
    df_cols = df_cols.div(df_cols.sum(axis=1), axis=0)

    df_midpoints_lst = [ 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i in range(len(df_cols.columns)):
        df_cols.iloc[:,i] = df_cols.iloc[:,i] * df_midpoints_lst[i]

    #calculate weighted average
    df_cols["Average num rooms"] = df_cols.sum(axis=1)

    #merge on index with original dataframe
    df_summary  = df_summary.merge(df_cols['Average num rooms'], how='outer', left_index=True, right_index=True)
    return  df_summary

def fuel_heat(df_summary):
    #Heating fuel
    df_cols = df_summary.rename(columns={'Estimate Total Utility gas':'Natural gas from underground pipes',
                                            'Estimate Total Bottled, tank, or LP gas': 'Propane (bottled gas)',
                                            'Estimate Total Electricity': 'Electricity',
                                            'Estimate Total Fuel oil, kerosene, etc.': 'Fuel oil',
                                            'Estimate Total Wood': 'Wood or pellets',
                                            'Estimate Total Solar energy':'solar_other',
                                            'Estimate Total Other fuel':'other_fuel_other',
                                            'Estimate Total Coal or coke':'coal_other',
                                            'Estimate Total No fuel used': 'Not applicable'})

    df_cols['Other_fuel'] = df_cols['solar_other'] + df_cols['other_fuel_other'] + df_cols['coal_other']

    #subset
    df_cols = df_cols[['Natural gas from underground pipes', 'Propane (bottled gas)', 'Electricity',
                          'Fuel oil', 'Wood or pellets','Other_fuel', 'Not applicable']]

    df_cols = df_cols.astype('int')

    #normalize by total number of HHs
    df_cols = df_cols.div(df_cols.sum(axis=1), axis=0)

    #merge on index qith original dataframe
    df_summary  = df_summary.merge(df_cols, how='outer', left_index=True, right_index=True)

    return  df_summary


def total_pop(df_summary):
    
    df_summary['total_population'] = df_summary['Estimate Total Female'] + df_summary['Estimate Total Male']
    
    return  df_summary
    
    
def subset_columns(df_summary):
    df_summary_final = df_summary[['state',
 'county',
 'tract',
 'Estimate Median year structure built',
 'Estimate Median number of rooms',
 'Median household income',
 'Child dependency ratio',
 'Old age dependency ratio',
 'Average household members',
 'Average household income',
 'Householder_binary',
 'Householder_female_level',
 'Mobile home',
 'Detached, 1 unit',
 'Attached, 1 unit',
 '2 to 4 units',
 '5 or more units',
 'Owner occupied',
 'Renter occupied',
 'Not Hispanic or Latino: White',
 'Not Hispanic or Latino: Black or African American',
 'Not Hispanic or Latino: American Indian and Alaska Native',
 'Not Hispanic or Latino: Asian alone',
 'Not Hispanic or Latino: Native Hawaiian and Other Pacific Islander',
 'Not Hispanic or Latino: Some other race alone',
 'Not Hispanic or Latino: Two or more races',
 'Hispanic or Latino',
 'Average year built',
 'Average num bedrooms',
 'Average num rooms',
 'Natural gas from underground pipes',
 'Propane (bottled gas)',
 'Electricity',
 'Fuel oil',
 'Wood or pellets',
 'Other_fuel',
 'Not applicable', 'total_population']]
    
    return  df_summary_final
    
