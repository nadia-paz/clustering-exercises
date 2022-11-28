############ IMPORTS ########
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from env import get_db_url

# display floats in human readable format
pd.options.display.float_format = '{:,.3f}'.format

################################

seed = 42


def acquire_mall_customers():
    query = 'SELECT * FROM customers'
    return pd.read_sql(query, get_db_url('mall_customers'))

def split_mall_customers(df):
    # create dummies
    df = pd.get_dummies(df, drop_first=True).drop(columns='customer_id')
    # split the data
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    return train, validate, test

def scale_mall_caustomers(train, validate, test):
    sc = StandardScaler()
    cols = ['age', 'annual_income']
    train[cols] = sc.fit_transform(train[cols])
    validate[cols] = sc.transform(validate[cols])
    test[cols] = sc.transform(test[cols])
    return train, validate, test

def get_mall_customers():
    df = acquire_mall_customers()
    train, validate, test = split_mall_customers(df)
    #train, validate, test = scale_mall_caustomers(train, validate, test)
    return train, validate, test
