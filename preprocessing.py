import pandas as pd
import glob
import numpy as np



# df_friday_afternoon_ddos = pd.read_csv("MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# df_friday_afternoon_portscan = pd.read_csv("MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
# df_Friday_morning = pd.read_csv("MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv")
# df_monday = pd.read_csv("MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv")
# df_thursday_afternoon = pd.read_csv("MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
# df_thurday_morning = pd.read_csv("MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
# df_tuesday = pd.read_csv("MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv")
# df_wednesday = pd.read_csv("MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv")

csv_files = glob.glob("MachineLearningCVE/*.csv")

df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

combined_df.to_csv("combined_dataset.csv", index=False)



# print(tabulate(df_friday_afternoon_ddos, headers='keys'))
def preprocess_dataframe(df):
    drop_columns = [
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
        'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'CWE Flag Count',
        'ECE Flag Count', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
        'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Down/Up Ratio', 'act_data_pkt_fwd', 'min_seg_size_forward'
    ]
    df.drop(columns=drop_columns, inplace=True, errors='ignore')

    #removes beginning and trailing white space
    df.columns = df.columns.str.strip()

    print("number of rows before dropping nulls: " + str(len(df)))
    # print("number of null values: ")
    # #checks each column for null values
    # print(df.isna().sum())
    # print('\nnan values below: ')
    # print(df.isnull().sum())
    # print('infinite values below: ')
    # count = np.isinf(df)
    # print(count)

    print(df.columns)
    unique_labels = df['Label'].unique()
    print(unique_labels)

    #REFERENCE OF ATTACK TO INTEGERS
    #BENIGN = 0
    #Infiltration = 1
    #Bot = 2
    #PortScan = 3
    #DDoS = 4
    #FTP-Patator = 5
    #SSH-Patator = 6
    #DoS slowloris = 7
    #DoS Slowhttptest = 8
    #DoS Hulk = 9
    #Dos GoldenEye = 10
    #Heartbleed = 11
    #Web Attack Brute Force = 12
    #Web Attack XSS = 13
    #Web Attack Sql Injection = 14

    label_mappings = {}
    i = 0
    for label in unique_labels:
        label_mappings[label] = i
        i += 1
    print(label_mappings)

    df['Label-Mappings'] = [label_mappings[label] for label in df['Label']]

    df.drop('Label', axis=1, inplace=True)

    #checks each column for null values
    print("number of null values: ")
    navalues = df.isna().sum()
    print(navalues)

    #making a dataframe with rows that have nan values
    mask_nullvalues = np.isnan(df[['Flow Bytes/s', 'Flow Packets/s']].any(axis=1))
    rows_with_nan_specific = df[df.isna().any(axis=1)]
    print("rows with nan values in flow bytes or flow packets: ")
    print(rows_with_nan_specific)

    # Determining inf values
    print('infinite values below: ')
    count = np.isinf(df).sum()
    print(count)
    #inf do exist, find where
    mask_specific = np.isinf(df[['Flow Bytes/s', 'Flow Packets/s']]).any(axis=1)
    rows_with_infs_specific = df[mask_specific]
    print("Rows with infinite values in 'flow bytes' or 'flow packets':")
    print(rows_with_infs_specific)

####################Not replacing inf with nan since many are DOS HULK, using clipping instead#########################
    #Replacing inf values with nan for imputation
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)

    #checking if inf are gone
    # print('infinite values below: ')
    # count = np.isinf(df).sum()
    # print(count)

    #determining the ratios of attack types in the data set and comparing to the ratio in the inf set
    attack_counts = df['Label-Mappings'].value_counts().sort_index()
    print("occurance count for each attack type in the normal dataset: ")
    print(attack_counts)

    inf_attack_counts = rows_with_infs_specific['Label-Mappings'].value_counts().sort_index()
    print("occurance count for each attack type in the infinite dataset: ")
    print(inf_attack_counts)

    nan_attack_types = rows_with_nan_specific['Label-Mappings'].value_counts().sort_index()
    print("occurance count for each attack type in the nan dataset: ")
    print(nan_attack_types)

    print("\n Checking how many label mappings there are: ")
    unique_labels_mappings_prior = df['Label-Mappings'].unique()
    print(unique_labels_mappings_prior)

    ###############I have determined that there is too much significance between null and inf values tied to DOS hulk attacks, so i will clip the values######

    #handle the nan values associated with dos hulk attacks, since a majority of nan values are on dos hulk, i will use their upper threshold calc, but the general one oon inf values
    dos_hulk_mask = df['Label-Mappings'] == 9
    columns_to_check = ['Flow Bytes/s', 'Flow Packets/s']

    for col in columns_to_check:
        finite_values = df.loc[dos_hulk_mask, col][np.isfinite(df.loc[dos_hulk_mask, col])]

        if not finite_values.empty:
            upper_threshold = finite_values.quantile(.99)

        df.loc[dos_hulk_mask, col] = df.loc[dos_hulk_mask, col].fillna(upper_threshold)
    print('the sum of null values in flow bytes and pakcets after removing them: ')
    print(df.loc[dos_hulk_mask, columns_to_check].isna().sum())

    ###rechecks how many nan values there are after removing those attached to dos hulk attacks
    rows_with_nan_specific_posthulk = df[df.isna().any(axis=1)]
    nan_attack_types_posthulk = rows_with_nan_specific_posthulk['Label-Mappings'].value_counts().sort_index()
    print("occurance count for each attack type in the nan dataset after removing those attached to DOS Hulk attacks: ")
    print(nan_attack_types_posthulk)
    ### only 409 records still contain null values, and they are all normal traffic, I will drop these records

    df.dropna(inplace=True)


    print("\n Checking how many label mappings there are: ")
    unique_labels_mappings_post = df['Label-Mappings'].unique()
    print(unique_labels_mappings_post)

    ####handle the inf values
    for col in df:
        if col == 'Label-Mappings':
            continue

        finite_values = df[col][np.isfinite(df[col])]

        if not finite_values.empty:
            # Define the lower and upper thresholds based on the finite values.
            lower_threshold = finite_values.quantile(0.01)
            upper_threshold = finite_values.quantile(0.99)

            df[col] = df[col].clip(lower=lower_threshold, upper=upper_threshold)
        else:
            print(f"Column '{col}' does not have any finite values to base thresholds on.")

    #verify that there are no infinite values remaining
    for col in df:
        num_infs = np.isinf(df[col]).sum()
        print(f"Column '{col}' still has {num_infs} infinite values.")

    print("\nAre there any null values in the data set (true is yes): ")
    isnan = np.isnan(df).any()
    print(isnan)
    print("\n Are there any inf values in the data set (true is yes): ")
    isinf = np.isinf(df).any()
    print(isinf)

    print("\n Checking how many label mappings there are: ")
    unique_labels_mappings_last = df['Label-Mappings'].unique()
    print(unique_labels_mappings_last)



    return df



processed_df = preprocess_dataframe(combined_df)

def get_preprocessed_df():
    return processed_df