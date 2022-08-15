import pandas as pd

# load csvs, interpret dates
df_train = pd.read_csv('train.csv', parse_dates=['Date'], dayfirst=True)
df_test = pd.read_csv('test.csv', parse_dates=['Date'], dayfirst=True)
df_stores = pd.read_csv('store.csv').replace('"', '', regex=True)


# method to convert date object to days offset from xmas (expected high sales point)
def calc_xmas_offset(date):
    xmas_ref = pd.Timestamp(f'{date.year}-12-24')
    return abs(pd.to_timedelta([date - xmas_ref])[0].days)


# all operations applied to both train and test set in a method for OOP purposes
def preprocess_common(data):
    # StateHoliday by default contains both "0" and 0, therefore string value is converted to create proper dummy values
    data.loc[data['StateHoliday'] == '0', 'StateHoliday'] = 0
    data = pd.get_dummies(data, columns=['StateHoliday'])

    data['WeekOfYear'] = data['Date'].dt.weekofyear

    # merge store set by store IDs present in both w inner join
    data = data.merge(on='Store', right=df_stores, how='inner')

    # Converting competition open to months, cut down on columns
    years_since_comp = data['Date'].dt.year - data['CompetitionOpenSinceYear']
    months_since_comp = data['Date'].dt.month - data['CompetitionOpenSinceMonth']
    data['CompetitionOpenForMonths'] = 12 * years_since_comp + months_since_comp

    # Same procedure, /4 weeks -> months
    years_since_promo = data['Date'].dt.year - data['Promo2SinceYear']
    weeks_since_promo = data['WeekOfYear'] - data['Promo2SinceWeek']
    data['Promo2RunningForMonths'] = 12 * years_since_promo + weeks_since_promo / 4

    # promo2 durations are 0 if promo2 is 0 too
    # a count of all missing values matches the amount of 0s for promo2, therefore no cross-reference needed
    data['Promo2RunningForMonths'] = data['Promo2RunningForMonths'].fillna(0)

    # value often not given despite competition distance being non-null;
    # as the parameter's impact is limited a mean can be assumed
    data['CompetitionOpenForMonths'] = data['CompetitionOpenForMonths'] \
        .fillna(data['CompetitionOpenForMonths'].mean())

    # original columns are no longer needed
    data = data.drop(['Promo2SinceWeek', 'Promo2SinceYear'], axis='columns')
    data = data.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis='columns')

    # convert remaining date object to useful (numeric) feature
    data['DaysToXmas'] = data['Date'].apply(calc_xmas_offset)

    # sort to 2013 -> 2015 so they can be passed to LSTM chronologically
    data = data.sort_values(by=['Store', 'Date'], ascending=True)

    return data


# 11 NaN values in test set (store 622 opening times), fill with 0 as clarified by competition host:
# https://www.kaggle.com/c/rossmann-store-sales/discussion/17048#96969
df_test['Open'] = df_test['Open'].fillna(0)

# low count, replace by mean
df_stores['CompetitionDistance'] = df_stores['CompetitionDistance'].fillna(df_stores['CompetitionDistance'].mean())

# one-hot-encoding for categorical values; the intervals fall within 3 distinct values so OHE relevant
df_stores = pd.get_dummies(df_stores, columns=['StoreType', 'Assortment', 'PromoInterval'])

# sets to apply common pre-processing to
sets = [df_train, df_test]
df_train, df_test = (preprocess_common(x) for x in sets)

# create a validation set from the last 6 weeks to match maximum prediction range; drop from train set afterwards
val_cutoff_date = pd.Timestamp(f'2015-06-19')
mask = (df_train['Date'] >= val_cutoff_date)

df_validation = df_train.loc[mask]
df_train = df_train[~df_train['Date'].isin(df_validation['Date'])]

# date and store no longer required, already sorted chronologically by store
df_train = df_train.drop(['Date', 'Store'], axis='columns')
df_test = df_test.drop(['Date', 'Store'], axis='columns')
df_validation = df_validation.drop(['Date', 'Store'], axis='columns')

# reveals that there are 899 entries ranging from 2013-15 per store id in the training set; used to cap model input
# for limited testing later on
# print(df_train.groupby(['Store']).size())

# save pre-processed dataset to save time
df_train.to_csv('train_clean.csv', index=False)
df_test.to_csv('test_clean.csv', index=False)
df_validation.to_csv('validation_clean.csv', index=False)
