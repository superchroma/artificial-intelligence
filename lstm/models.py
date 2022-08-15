import csv
from pathlib import Path

import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 899 entries per store; use multiple (*20) for better results in testing; hardware limitations
df_train = pd.read_csv('train_clean.csv', nrows=17980)
df_test = pd.read_csv('test_clean.csv')
df_validation = pd.read_csv('validation_clean.csv', nrows=17980)

predictors = ['DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'StateHoliday_0',
              'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c', 'WeekOfYear', 'CompetitionDistance', 'Promo2',
              'StoreType_a', 'StoreType_b', 'StoreType_c', 'StoreType_d', 'Assortment_a', 'Assortment_b',
              'Assortment_c',
              'PromoInterval_Feb,May,Aug,Nov', 'PromoInterval_Jan,Apr,Jul,Oct',
              'PromoInterval_Mar,Jun,Sept,Dec', 'CompetitionOpenForMonths', 'Promo2RunningForMonths', 'DaysToXmas']

# don't include columns not represented in test set to achieve uniform input shape
predictors = [x for x in predictors if x in df_test.columns]

# convert to numpy arrays and extract target values
x_train = df_train.loc[:, predictors].values
x_test = df_test.loc[:, predictors].values
x_validation = df_validation.loc[:, predictors].values
y_validation = df_validation.loc[:, ['Sales']].values
y_train = df_train.loc[:, ['Sales']].values

# scale to stdv = 1 mean 0, keep scaler instance as it has to be applied to the test set too to avoid overfitting
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

# apply PCA to reduce dimensionality
pca = PCA(n_components=0.95)
pca_train = pca.fit_transform(x_train)
pca_test = pca.transform(x_test)
pca_val = pca.transform(x_validation)

# concat back to dataframe and re-attach sales column for predictions
df_pca = pd.DataFrame(data=pca_train,
                      columns=['component ' + str(x) for x in range(0, len(pca_train[1]))])
df_pca = pd.concat([df_pca, df_train[['Sales']]], axis=1)

# manual verification of PCA component sets
print(df_pca.head())

pca_np = df_pca.to_numpy()
# LSTM expects shape (samples, timesteps, features)
lstm_train = pca_train.reshape((pca_train.shape[0], 1, pca_train.shape[1]))

model = Sequential()
# specify dimensions again, tuple as sample count is automatically assumed to be 1+
model.add(LSTM(32, input_shape=(1, lstm_train.shape[2])))
# Dropout to avoid overfitting, especially when using limited data size (hardware constraints training with full input)
model.add(Dropout(0.2))
# predictions for 1 day each
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

# reshape pca data to fit LSTM required shape
pca_val = pca_val.reshape(pca_val.shape[0], 1, pca_val.shape[1])
y_validation = y_validation.reshape(y_validation.shape[0], 1, y_validation.shape[1])

stored_weights = Path("final_weights.h5")
# if weights exist, load them
if stored_weights.is_file():
    print("Loading stored weights")
    model.load_weights('final_weights.h5')
else:
    model.fit(lstm_train, y_train, epochs=120, batch_size=7, verbose=2
              , validation_data=(pca_val, y_validation))
model.save_weights('final_weights.h5')


# weekly chunks as batch size is 7 days, validation set is also ordered chronologically
def split_time_chunks(data, chunks):
    return [data[x:x + chunks] for x in range(0, len(data), chunks)]


# manual sample showcase of LSTM experiment
weekly_observations = split_time_chunks(pca_val, 7)

days_offset = 0
# for week in weekly_observations:
#    # get actual values for the given time frame for comparison
#    week_y = y_validation[days_offset:days_offset + 7]
#    days_offset += 7
#    # already in 3D shape
#    y_pred = model.predict(week, verbose=0)
#    print("PREDICTED", y_pred)
#    print("ACTUAL", week_y)


# predict test set
pca_test = pca_test.reshape(pca_test.shape[0], 1, pca_test.shape[1])
test_weeks = split_time_chunks(pca_test, 7)
ids = df_test.loc[:, ['Id']].values
final_preds = {}
id_count = 0

for week in test_weeks:
    print("row", id_count, "/", len(ids))
    for y_pred in model.predict(week):
        final_preds[ids[id_count][0]] = y_pred[0]
        id_count += 1

# write to submission file (not original order as it is grouped by id and date; hopefully okay?)
header = ["Id", "Sales"]
with open('predictions.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(final_preds.items())
