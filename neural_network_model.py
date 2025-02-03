import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
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

#fixme: (log normalization did not work due to negative values)
#data normalization (Tried log normalization, but due to too extreme of values, will not work with scaling methods
#second try will be to use Yeo-Johnson Transformation for normalization
# X_train_log = np.log1p(X_train)
# X_test_log = np.log1p(X_test)

pt = PowerTransformer(method='yeo-johnson')
X_train_normalized = pt.fit_transform(X_train)
X_test_normalized = pt.fit_transform(X_test)

#fixme: (testing only) ensures dataframe has no inf or nan or negatives
print("\nAre there any null values in the data set (true is yes): ")
isnan = np.isnan(df).any()
print(isnan)
print("\n Are there any inf values in the data set (true is yes): ")
isinf = np.isinf(df).any()
print(isinf)
any_negatives = df.lt(0).any().any()
print("are there any negatives?")
print(any_negatives)

#fixme: (remove once scaling works) trying scaling
# X_train_scaled = min_max_scaling(X_train)
# X_test_scaled = min_max_scaling(X_test)

##Scaling begins
X_train_scaled = standard_scaling(X_train_normalized)
X_test_scaled = standard_scaling(X_test_normalized)
#################Above is standard scaling, below is min max, test which is better#######################
# X_train_scaled = min_max_scaling(X_train_normalized)
# X_test_scaled = min_max_scaling(X_test_normalized)

#Applying SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced, = smote.fit_resample(X_train_scaled, y_train)


#converts the normalized and scaled back to dataframes
X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=X_train.columns)
X_test_balanced_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print('Yeo Johnson-normalized Training Data: ')
print(X_train_balanced_df.head())

print('\Yeo Johnson-normalized Test data: ')
print(X_test_balanced_df.head())

print("Original training set class distribution:")
print(y_train.value_counts())

print("\nBalanced training set class distribution:")
print(pd.Series(y_train_balanced).value_counts())




