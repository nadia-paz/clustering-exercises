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
    total: 14 rows with ids were removed
    '''
    #df.drop(columns=['id', 'parcelid'], inplace=True)
    
    # drop the columns that consist only of null values
    for col in df.columns.tolist():
        if df[col].isna().sum() == df.shape[0]:
            df.drop(columns=col, inplace=True)
    # based on value_counts() i would keep buildingqualitytypeid and heatingorsystemtypeid

    # rename those columns to remove id from their name
   # df.rename(columns={'buildingqualitytypeid':'building_quality_type', 
               # 'heatingorsystemtypeid':'heating_system_type'}, inplace=True)
    # remove columns that are  different ids
    columns = []
    for col in df.columns.tolist():
        if col.endswith('id'):
            columns.append(col)
    df.drop(columns=columns, inplace=True)

def fill_nulls(df):
    '''
    this function accept a zillow data frame as a parameter
    makes the following changes to the columns:
    - fills null values with 0 for the columns where it is reasonable
    - renames columns to human readable format
    - removes columns where too many null values, replacing is not logical, 
        too many or only one categorical variable(s), 
        columns that are identical to other columns
    in total:
        - in 10 columns null values were replaced with 0
        - 13 columns were dropped
    '''
    # potentially to be dropped because of the high # of NaN values:
    # those columns I don't fill with nulls
    # finishedfloor1squarefeet, finishedsquarefeet50, finishedsquarefeet6, poolsizesum, propertyzoningdesc
    # yardbuildingsqft17, yardbuildingsqft26, fireplaceflag, taxdelinquencyflag, airconditioningdesc
    # architecturalstyledesc, storydesc, typeconstructiondesc

    # replace NaN with zeros
    #df['pools'] = df.pools.replace({np.NAN:0})
    #df.basementsqft = df.basementsqft.fillna(0) has 52K+ null values
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
    #df.heatingorsystemdesc = df.heatingorsystemdesc.fillna('None') # check if it is ok


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
            'regionidcity':'city_id',
            'regionidzip':'zip',
            'unitcnt':'unit',
            'yearbuilt':'year_built',
            'structuretaxvaluedollarcnt':'structure_price',
            'taxvaluedollarcnt':'price',
            'landtaxvaluedollarcnt':'land_price',
            'taxamount':'tax_amount',
            }, inplace=True)

    # too many  or 1 categorical unique values or identical to other columns
    df.drop(columns=['calculatedbathnbr', 'basementsqft', 'finishedsquarefeet12',
                    'rawcensustractandblock', 
                    'regionidcounty', 'regionidneighborhood', 'roomcnt',
                    'censustractandblock', 'assessmentyear', 'transactiondate',
                    'propertylandusedesc', 'heatingorsystemdesc'], 
            inplace=True)
    # drop fullbath as almost identical to bathcount
    df.drop(columns='fullbath', inplace=True)


#Define function to drop columns/rows based on proportion of nulls
def drop_nulls(df, prop_required_column=0.75, prop_required_row=0.75):
    '''
    - the function accepts a zillo data frame,
    percentage of min values in columns and rows 
    - drops duplicates
    - drops columns pool_10, pool_2, pool_7
    - drops all columns and rows where the number of nulls is way too big
    - drops other nulls
    in total drops: 19 columns and 1521 rows
    '''
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
    '''
    the function accepts a zillow data frame as a parameter
    returns a data frame without some outliers
    in total removes 1208 rows
    '''
    # zip code out of max range
    df = df[df.zip <= 99950]
    # remove bedrooms and bathrooms > 0 and < 7
    df = df[df.bath != 0]
    df = df[df.beds != 0]
    df = df[df.beds < 7]
    df = df[df.bath < 7]
    # remove sq feet below 300 and above 6_000
    df = df[df.sqft >= 300]
    df = df[df.sqft <= 6_000]

    # target variable
    # removes logerror < -0.55 and > 0.55
    q1 = - 0.55
    q3 = 0.55
    #q1 = df.logerror.quantile(0.01)
    #q3 = df.logerror.quantile(0.99)
    df = df[(df.logerror > q1) & (df.logerror < q3)] # removes 1034 rows


    return df

def transform_columns(df):
    
    # change floats to ints
    for col in df.iloc[:, :-3].columns:
        if df[col].dtype != 'object':
            df[col] = df[col].astype(int)
    # remove bath where bath != fullbath
    df = df[df.bath == df.fullbath] # removes 94 rows
    # drop fullbath as identical
    df.drop(columns='fullbath', inplace=True)
    # create a list of numerical columns
    numerical_columns = ['sqft', 'garage_sqft', 'latitude', 'longitude', 
                    'lot_sqft', 'year_built',
                    'structure_price', 'price', 'land_price', 
                    'tax_amount', 'logerror']
    # change not numerical columns to categories
    for col in df.columns:
        if col not in numerical_columns:
            df[col] = pd.Categorical(df[col])
    
    return df

def transform_columns2(df):
    '''
    the function accept zillow data frame as a parameter
    transforms:
    --> most of floats (exc tax amout and logerror) into integer
    --> columns with small numerical values - to 'uint8' data type
    --> county_land_code and fips to categories
    returns a data frame with transformed values
    '''
    
    # change floats to ints
    for col in df.iloc[:, :-2].columns:
        if df[col].dtype != 'object':
            df[col] = df[col].astype(int)

    # create a list of numerical columns
    numerical_columns = ['sqft', 'garage_sqft', 'latitude', 'longitude', 
                    'lot_sqft', 'year_built', 'city_id', 'zip',
                    'structure_price', 'price', 'land_price', 
                    'tax_amount', 'logerror']
    # change not numerical columns to categories
    for col in df.columns:
        if col not in numerical_columns:
            if col in ['county_land_code', 'fips']:
                df[col] = pd.Categorical(df[col])
            else:
                df[col] = df[col].astype('uint8')
    
    return df

def engineering(df):
    '''
    the function accepts zillow data frame as a parameter
    creates a new column age = 2017 - year_built
    creates a new column bed_bath_ratio = bed / bath
    creates a new column count_name based on fips
    rearranges the order of columns
    '''
    df['age'] = 2017 - df.year_built
    df['bed_bath_ratio'] = round(df.beds / df.bath, 2)
    
    # add a new column with county names
    df['county_name'] = np.select([(df.fips == 6037), (df.fips == 6059), (df.fips == 6111)],
                             ['LA', 'Orange', 'Ventura'])
    df.drop(columns=['year_built', 'fips'], inplace=True)
    # column to category data type
   
    new_order_cols = ['sqft',  'garage_sqft', 'lot_sqft', 'age', 
        'structure_price', 'price','land_price', 'tax_amount', 
        'bed_bath_ratio', 'city_id', 'zip', 'latitude', 'longitude',
        'bath', 'beds', 'fireplace', 'garage', 'hottub_spa', 'pool', 
        'unit', 'county_land_code', 'county_name', 'logerror']
    return df[new_order_cols]

######## get_zillow ready for exploration ######

def get_zillow():
    df = acquire_zillow()
    clean_from_ids(df)
    fill_nulls(df)
    drop_nulls(df)
    df = handle_outliers(df)
    df = transform_columns(df)
    df = engineering(df)

    filename = 'clean_zillow.csv'
    df.to_csv(filename, index_label = False)

    return df

def get_zillow2():
    df = acquire_zillow()
    clean_from_ids(df)
    fill_nulls(df)
    drop_nulls(df)
    df = handle_outliers(df)
    df = transform_columns2(df)
    df = engineering(df)

    filename = 'clean_zillow.csv'
    df.to_csv(filename, index_label = False)

    return df


############### SPLIT FUCNTIONS ########
def split_zillow(df):
    '''
    This function takes in a dataframe and splits it into 3 data sets
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    #split_db class verision with random seed
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    return train, validate, test

def full_split3_zillow(train, validate, test, target):
    '''
    accepts train, validate, test data sets and the name of the target variable as a parameter
    splits the data frame into:
    X_train, X_validate, X_test, y_train, y_validate, y_test
    '''
    #train, validate, test = train_validate_test_split(df, target)

    #save target column
    y_train = train[target]
    y_validate = validate[target]
    y_test = test[target]

    #remove target column from the sets
    train.drop(columns = target, inplace=True)
    validate.drop(columns = target, inplace=True)
    test.drop(columns = target, inplace=True)

    return train, validate, test, y_train, y_validate, y_test



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
