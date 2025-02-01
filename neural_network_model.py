import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from preprocessing import get_preprocessed_df
from imblearn.over_sampling import SMOTE

df = get_preprocessed_df()

def standard_scaling(xlog):
    scaler = StandardScaler()
    xlog_scaled = scaler.fit_transform(xlog)
    return xlog_scaled

def min_max_scaling(xlog):
    min_max_scaler = MinMaxScaler()
    xlog_scaled = min_max_scaler.fit_transform(xlog)
    return xlog_scaled

#train-test splits
X = df.drop('Label-Mappings', axis=1)
y = df['Label-Mappings']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42
)

#data normalization
X_train_log = np.log1p(X_train)
X_test_log = np.log1p(X_test)

##Scaling begins
# X_train_scaled = standard_scaling(X_train_log)
# X_test_scaled = standard_scaling(X_test_log)
#################Above is standard scaling, below is min max, test which is better#######################
X_train_scaled = min_max_scaling(X_train_log)
X_test_scaled = min_max_scaling(X_test_log)

#Applying SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced, = smote.fit_resample(X_train_scaled, y_train)


#converts the normalized and scaled back to dataframes
X_train_log_scaled = pd.DataFrame(X_train_log, columns=X_train.columns)
X_test_log_scaled = pd.DataFrame(X_test_log, columns=X_test.columns)

print('Log-normalized Training Data: ')
print(X_train_log_scaled.head())

print('\nLog-normalized Test data: ')
print(X_test_log_scaled.head())

print("Original training set class distribution:")
print(y_train.value_counts())

print("\nBalanced training set class distribution:")
print(pd.Series(y_train_balanced).value_counts())


