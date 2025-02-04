import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from tensorflow.keras import Sequential
from imblearn.over_sampling import ADASYN, BorderlineSMOTE

from preprocessing import get_preprocessed_df
from imblearn.over_sampling import SMOTE
import tensorflow as tf
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix

df = get_preprocessed_df()

def standard_scaling(xtrain, xtest):
    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    return xtrain_scaled, xtest_scaled

def min_max_scaling(xtrain, xtest):
    min_max_scaler = MinMaxScaler()
    xtrain_scaled = min_max_scaler.fit_transform(xtrain)
    xtest_scaled = min_max_scaler.transform(xtest)
    return xtrain_scaled, xtest_scaled

######### Printing all label categories #########
unique_labels = df['Label-Mappings'].unique()
print(unique_labels)

############ SPLITTING DATA ###############
#train-test splits
X = df.drop('Label-Mappings', axis=1)
y = df['Label-Mappings']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42
)


########### NORMALIZATION METHODS ############
#LOG NORMALIZATION
#fixme: (log normalization did not work due to negative values)
#data normalization (Tried log normalization, but due to too extreme of values, will not work with scaling methods
#second try will be to use Yeo-Johnson Transformation for normalization
# X_train_log = np.log1p(X_train)
# X_test_log = np.log1p(X_test)

# YEO JOHNSON NORMALIZATION
pt = PowerTransformer(method='yeo-johnson')
X_train_normalized = pt.fit_transform(X_train)
X_test_normalized = pt.transform(X_test)


################### PRINT TESTING ####################
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


################### Scaling begins #################
X_train_scaled, X_test_scaled = standard_scaling(X_train_normalized,X_test_normalized)
# Above is standard scaling, below is min max, test which is better#######################
# X_train_scaled, X_test_scaled = min_max_scaling(X_train_normalized,X_test_normalized)


###############Balancing Methods that I am Trying####################
### SMOTE BALANCING
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced, = smote.fit_resample(X_train_scaled, y_train)

# ADASYN BALANCING
# high performance for 0,3,4,5,8,9,10
# low precision and high recall on 2, 12
# low precision and recall on 13,14
# very low support for 1,11
# adasyn = ADASYN(random_state=42)
# X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train_scaled, y_train)


############## Conversion into Dataframe ##########################
#converts the normalized and scaled back to dataframes
X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=X_train.columns)
X_test_balanced_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)


########### PRINTING THE DATASETS #####################
print('\n FINALIZED Training Data: ')
print(X_train_balanced_df.head())

print('\n FINALIZED Test data: ')
print(X_test_balanced_df.head())

print("Original training set class distribution:")
print(y_train.value_counts())

print("\nBalanced training set class distribution:")
print(pd.Series(y_train_balanced).value_counts())


###########Split into Training and Validation Sets ###########
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_balanced, y_train_balanced, test_size=0.2, random_state=42
)


############## Build the Neural Network Model ##############
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_final.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(15, activation='softmax')
])


############ Compile the model ######################
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


############## Train the Model #####################
history = model.fit(
    X_train_final, y_train_final,
    epochs=5,
    batch_size=32,
    validation_data=(X_val, y_val)
)

model.save('second_working.keras')


########## Evaluate the Model on the Test Set ################
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# Detailed Evaluation
# Generate predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)


# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


