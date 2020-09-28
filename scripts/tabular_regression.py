# %%
import numpy as np 
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

import sys
sys.path.append('../scripts/') 

from aux_funcs import label_encoder, load_ct_scans, competition_metric, extract_img_id, get_config, make_ct_git, filter_by_std
from sklearn.model_selection import train_test_split

config = get_config()
train_df = pd.read_csv(config['data_path']+'/train.csv')
le_patient = label_encoder()
le_patient.fit(np.unique(train_df.Patient).tolist())


# %%
le_patient = label_encoder()
le_patient.fit(np.unique(train_df.Patient).tolist())
sex_encoding = {'Male':0, 'Female':1}

train_df.Patient = train_df.Patient.apply(lambda x: le_patient.transform([x]))
train_df.Sex = train_df.Sex.apply(lambda x: 1 if x == 'Female' else 0)
train_df['current_smoker'] = train_df.SmokingStatus.apply(lambda x: 1 if x == 'Currently smokes' else 0)
train_df['ex_smoker'] = train_df.SmokingStatus.apply(lambda x: 1 if x == 'Ex-smoker' else 0)
train_df['never_smoker'] = train_df.SmokingStatus.apply(lambda x: 1 if x == 'Never smoked' else 0)
train_df.drop(['SmokingStatus'],1,inplace=True)

train_df.Percent /= 100


# %%
patients_train, patients_test = train_test_split(np.unique(train_df['Patient'].values), test_size=0.3, shuffle=True, random_state=63)


# %%
df_train = []
for p in patients_train:
    mask = train_df['Patient'] == p
    df_train.append(train_df.values[mask])
df_train = pd.DataFrame(np.concatenate(df_train, 0), columns=list(train_df.columns))

df_test = []
for p in patients_test:
    mask = train_df['Patient'] == p
    df_test.append(train_df.values[mask])
df_test = pd.DataFrame(np.concatenate(df_test, 0), columns=list(train_df.columns))


# %%
initial_week = -12
final_week = 133

n_patients = np.unique(train_df['Patient'].values).shape[0]
n_weeks = final_week - initial_week

fvc_mat = np.ones((n_patients, n_weeks))*-1

for x in np.unique(train_df['Patient'].values):
    mask = train_df['Patient'].values == x
    fvcs = train_df['FVC'].values[mask]
    weeks = train_df['Weeks'].values[mask]
    for i in range(weeks.shape[0]):
        fvc_mat[x,weeks[i]] = fvcs[i]


# %%
def format_data(df):
    data = []
    for i in range(-155,155):
        if i == 0:
            continue
        new_data = df.copy()
        new_data['delta_week'] = new_data.groupby('Patient')['Weeks'].diff(i)
        new_data['new_fvc'] = new_data.groupby('Patient')['FVC'].shift(i)
        new_data = new_data.dropna()
        if new_data.shape[0] == 0:
            continue
        data.append(new_data.values)
    data = pd.DataFrame(np.concatenate(data,0), columns=['Patient','Weeks','FVC','Percent','Age','Sex','current_smoker','ex_smoker','never_smoker','delta_week','new_fvc'])

    X = data.dropna()[['delta_week', 'Weeks', 'FVC', 'Percent', 'Age', 'Sex','current_smoker','ex_smoker','never_smoker']]
    Y = data.dropna()[['new_fvc']]

    return X.values, Y.values.squeeze()


# %%
X_train, Y_train = format_data(df_train)
X_test, Y_test = format_data(df_test)


# %%
# Having a look only at the pre and post fvc

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

n_estimators = np.linspace(100,5000, 10, dtype=np.int32)
max_depth = np.linspace(2,X_train.shape[1],5, dtype=np.int32)
criteria = np.array(['mse','mae'])
eval_configs = np.array(np.meshgrid(n_estimators, max_depth, criteria)).T.reshape(-1,3)

cv_results = []

for config in tqdm(eval_configs):
    regr = RandomForestRegressor(n_estimators=int(config[0]), max_depth=int(config[1]), random_state=0, criterion=str(config[2]), n_jobs=-1)
    scores = cross_validate(regr, X_train, Y_train, scoring=['neg_mean_squared_error'], cv=3, return_train_score=True, n_jobs=-1)
    cv_results.append(scores['test_neg_mean_squared_error'])
cv_results = np.stack(cv_results,0)


# %%
cv_results


# %%
'''
regr = RandomForestRegressor(n_estimators=int(opt_config[0]), max_depth=int(opt_config[1]), random_state=0, criterion=str(opt_config[2]), n_jobs=-1)
regr.fit(X_train, Y_train)

mu = np.mean(np.stack([x.predict(X_train) for x in regr.estimators_], 0), 0)
sigma = np.std(np.stack([x.predict(X_train) for x in regr.estimators_], 0), 0)

print('Error new_fvc', np.mean(competition_metric(Y_train, mu, sigma)))'''


