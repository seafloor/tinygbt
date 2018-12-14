import pandas as pd
import numpy as np
from tinygbt import Dataset, GBT
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print('Load data...')
df = pd.read_hdf('../regression_snp_example.hdf5')
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:].values, df.iloc[:, 0].values,
                                                    test_size=0.25,
                                                    random_state=1234,
                                                    shuffle=True)
train_data = Dataset(X_train, y_train)
eval_data = Dataset(X_test, y_test)

beta_df = pd.read_csv('../regression_example_betas.txt', sep='\t')
prs_betas = np.log(beta_df['or'].values)
params = {'prs_betas': prs_betas}

print('Start training...')
gbt = GBT()
gbt.train(params,
          train_data,
          num_boost_round=20,
          valid_set=eval_data,
          early_stopping_rounds=20)

print('Start predicting...')
y_pred = []
for x in X_test:
    y_pred.append(gbt.predict(x, num_iteration=gbt.best_iteration))

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
