import pandas as pd

""""
Function that returns test and train data, based on the indices in the buckets_location .csv file
"""
def cross_validation_train_test_split(data_set, buckets_location, bucket_number):
    # Read all the bucket indices for the specified bucket number from the provided CSV file
    bucket_indices = pd.read_csv(buckets_location).iloc[bucket_number].dropna().values

    # Split the entire data set into testing and training data
    data_test = data_set.iloc[bucket_indices]
    data_train = data_set.drop(bucket_indices, axis='index')

    # Reset the indices of the data
    data_test.reset_index(drop=True, inplace=True)
    data_train.reset_index(drop=True, inplace=True)

    return data_train, data_test

