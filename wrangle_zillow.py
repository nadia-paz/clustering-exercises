import os
import pandas as pd
import numpy as np

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.linear_model import LinearRegression

from env import get_db_url

########## GLOBAL VARIABLES ##########

# random state seed
seed = 2912

# sql query to get the data
#Create the SQL query
query = '''
        SELECT prop.*,
        predictions_2017.logerror,
        predictions_2017.transactiondate,
        air.airconditioningdesc,
        arch.architecturalstyledesc,
        build.buildingclassdesc,
        heat.heatingorsystemdesc,
        land.propertylandusedesc,
        story.storydesc,
        type.typeconstructiondesc
        FROM properties_2017 prop
        JOIN (
            SELECT parcelid, MAX(transactiondate) AS max_transactiondate
            FROM predictions_2017
            GROUP BY parcelid
            ) pred USING(parcelid)
        JOIN predictions_2017 ON pred.parcelid = predictions_2017.parcelid
                          AND pred.max_transactiondate = predictions_2017.transactiondate
        LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        LEFT JOIN storytype story USING(storytypeid)
        LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
        WHERE propertylandusedesc = "Single Family Residential"
            AND transactiondate <= '2017-12-31'
            AND prop.longitude IS NOT NULL
            AND prop.latitude IS NOT NULL
        '''

def acquire_zillow():
    '''
    acuires data from codeup data base
    returns a pandas dataframe with
    'Single Family Residential' properties of 2017
    from zillow
    '''
    
    filename = 'zillow.csv'

    url = get_db_url('zillow')
    
    # if csv file is available locally, read data from it
    if os.path.isfile(filename):
        df = pd.read_csv(filename) 
    
    # if *.csv file is not available locally, acquire data from SQL database
    # and write it as *.csv for future use
    else:
        # read the SQL query into a dataframe
        df =  pd.read_sql(query, url)
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index_label = False)
        
    return df

def clean_from_ids(df):
    '''
    the function accepts a dataframe as a parameter
    goes through all columns and removes the ones 
    that end with id
    '''
    #df.drop(columns=['id', 'parcelid'], inplace=True)
    
    # drop the columns that consist only of null values
    for col in df.columns.tolist():
        if df[col].isna().sum() == df.shape[0]:
            df.drop(columns=col, inplace=True)
    # based on value_counts() i would keep buildingqualitytypeid and heatingorsystemtypeid

    # rename those columns to remove id from their name
    df.rename(columns={'buildingqualitytypeid':'building_quality_type', 
                'heatingorsystemtypeid':'heating_system_type'}, inplace=True)
    # remove columns that are  different ids
    columns = []
    for col in df.columns.tolist():
        if col.endswith('id'):
            columns.append(col)
    df.drop(columns=columns, inplace=True)

def fill_nulls(df):
    # potentially to be dropped because of the high # of NaN values:
    # finishedfloor1squarefeet, finishedsquarefeet50, finishedsquarefeet6, poolsizesum, propertyzoningdesc
    # yardbuildingsqft17, yardbuildingsqft26, fireplaceflag, taxdelinquencyflag, airconditioningdesc
    # architecturalstyledesc, storydesc, typeconstructiondesc

    # replace NaN with zeros
    df.basementsqft = df.basementsqft.fillna(0)
    df.fireplacecnt = df.fireplacecnt.fillna(0)
    df.fullbathcnt = df.fullbathcnt.fillna(0)
    df.garagecarcnt = df.garagecarcnt.fillna(0)
    df.garagetotalsqft = df.garagetotalsqft.fillna(0)
    df.hashottuborspa = df.hashottuborspa.fillna(0)
    df.poolcnt = df.poolcnt.fillna(0)
    df.pooltypeid10 = df.pooltypeid10.fillna(0)
    df.pooltypeid2 = df.pooltypeid2.fillna(0)
    df.pooltypeid7 = df.pooltypeid7.fillna(0)
    df.unitcnt = df.unitcnt.fillna(0)
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna('None') # check if it is ok

    df.rename(columns={
            'calculatedfinishedsquarefeet':'sqft',
            'bathroomcnt':'bath',
            'bedroomcnt':'beds',
            'fireplacecnt':'fireplace', 
            'fullbathcnt':'fullbath',
            'garagecarcnt':'garage',
            'garagetotalsqft':'garage_sqft',
            'hashottuborspa':'hottub_spa',
            'lotsizesquarefeet': 'lot_sqft',
            'poolcnt':'pool',
            'pooltypeid10':'pool_10',
            'pooltypeid2':'pool_2',
            'pooltypeid7':'pool_7',
            'propertycountylandusecode':'county_land_code',
            'unitcnt':'unit',
            'yearbuilt':'year_built',
            'structuretaxvaluedollarcnt':'structure_price',
            'taxvaluedollarcnt':'price',
            'landtaxvaluedollarcnt':'land_price',
            'taxamount':'tax_amount',
            'heatingorsystemdesc':'heating_system'
            }, inplace=True)

    # too many  or 1 categorical unique values or identical to other columns
    df.drop(columns=['calculatedbathnbr', 'basementsqft', 'finishedsquarefeet12', 
                    'rawcensustractandblock', 'regionidcity', 'regionidcounty', 'regionidneighborhood', 
                    'censustractandblock', 'assessmentyear', 'transactiondate',
                    'regionidzip', 'propertylandusedesc'], 
            inplace=True)


#Define function to drop columns/rows based on proportion of nulls
def drop_nulls(df, prop_required_column=0.75, prop_required_row=0.75):

    df.drop_duplicates(inplace=True)
    
    # assign 1 to pools where pool_10=1 and pool=0
    df.pool = np.where((df.pool == 0) & (df.pool_10 == 1), 1, df.pool)
    df.drop(columns=['pool_10', 'pool_2', 'pool_7'], inplace=True)
    
    prop_null_column = 1 - prop_required_column
    
    for col in list(df.columns):
        
        null_sum = df[col].isna().sum()
        null_pct = null_sum / df.shape[0]
        
        if null_pct > prop_null_column:
            df.drop(columns=col, inplace=True)
            
    row_threshold = int(prop_required_row * df.shape[1])
    
    df.dropna(axis=0, thresh=row_threshold, inplace=True)
    df.dropna(axis=0, inplace=True)

def handle_outliers(df):
    df = df[df.sqft >= 300]
    df = df[df.bath != 0]
    df = df[df.beds != 0]
    return df

def transform_columns(df):
    
    # change floats to ints
    for col in df.iloc[:, :-3].columns:
        if df[col].dtype != 'object':
            df[col] = df[col].astype(int)
    # create a list of numerical columns
    numerical_columns = ['sqft', 'garage_sqft', 'latitude', 'longitude', 'lot_sqft', 'year_built',
                    'structure_price', 'price', 'land_price', 'tax_amount', 'logerror']
    # change not numerical columns to categories
    for col in df.columns:
        if col not in numerical_columns:
            df[col] = pd.Categorical(df[col])
    
    return df

######## get_zillow ready for exploration ######

def get_zillow():
    df = acquire_zillow()
    clean_from_ids(df)
    fill_nulls(df)
    drop_nulls(df)
    df = handle_outliers(df)
    df = transform_columns(df)

    filename = 'clean_zillow.csv'
    df.to_csv(filename, index_label = False)
    
    return df


############ printing functions ###########

def null_counter(df):
    new_columns = ['name', 'num_rows_missing', 'pct_rows_missing']
    new_df = pd.DataFrame(columns=new_columns)
    for i, col in enumerate(list(df.columns)):
        num_missing = df[col].isna().sum()
        pct_missing = num_missing / df.shape[0]
        
        new_df.loc[i] = [col, num_missing, pct_missing]
    
    return new_df

def print_value_counts(df):
    for col in df.columns.tolist():
        print(col)
        display(df[col].value_counts(dropna=False).reset_index())
        print()
