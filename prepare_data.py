import pandas as pd

def prepare_data(data, splits=0.8):
    '''
    Prepares the data by splitting into training, validation(if specified) and test sets
    
    Parameters
    ----------
    data : pandas.core.frame.DataFrame
          Dataset that needs to be splitted
          Dates in the dataset should be an index with type as pandas.core.indexes.datetimes.DatetimeIndex 
          and needs to be sorted
    splits : float, tuple
             Default = 0.8; splitting data in two sets - train and test of size 0.8 and 0.2 respectively
             float: Splits in two sets - train and test
                    Only accepts value between range (0.0, 1.0)
             tuple(train_size, validation_size): Splits in three sets - train, validation and test
                                                 train_size -> Train size
                                                 validation_size -> Validation size
                                                 Should only contain values between range (0.0, 1.0) and 
                                                 "train_size + test_size" should also be between (0.0, 1.0)
                                           
    Returns
    -------
    Train, test and validation(if tuples) splits : (pandas.core.frame.DataFrame)
    If splits is float - Train and test splits 
    If splits is a tuple - Train, validation, and test splits
    '''
    
    # if data is not pandas dataframe then error
    if not isinstance(data, pd.core.frame.DataFrame):
        raise TypeError("data is not a pandas dataframe")
    
    if not isinstance(data.index, pd.core.indexes.datetimes.DatetimeIndex):
        raise TypeError("data does not have index in pandas DatetimeIndex format")
    
    # if splits is not float or tuple then error
    if not isinstance(splits, (float, tuple)):
        raise TypeError("splits is not float or tuple")

    # check if dates are sorted or not, and if not raise exception
    if(data.index.is_monotonic_increasing==False):
        raise Exception("Dates are not sorted in the dataframe")
    
    #if splits is float then perform
    if isinstance(splits, float):
        # check if splits is between 0.0 and 1.0
        if(splits<=0.0 or splits>=1.0):
            raise Exception("splits can be only between 0.0 and 1.0")
        
        # splits in two sets - train and test and returns
        train = data[:int(splits*(len(data)))]
        test = data[int(splits*(len(data))):]
        return train, test 

    #if splits is tuple then perform
    if isinstance(splits, tuple):
        train_size = splits[0]
        test_size = splits[1]
        # check if tuple vals contain float
        if not isinstance(train_size, (float)):
            raise TypeError("Should contain only float")
        if not isinstance(test_size, (float)):
            raise TypeError("Should contain only float")
            
        # tuple len is not equal to two then exception
        if len(splits)!=2:
            raise Exception("Split should contain only two values")
            
        # check if all value in splits is between 0.0 and 1.0
        if(train_size<=0.0 or train_size>=1.0):
            raise Exception("splits can only be between 0.0 and 1.0")
        if(test_size<=0.0 or test_size>=1.0):
            raise Exception("splits can only be between 0.0 and 1.0")
        
        # check if sum of splits is between 0.0 and 1.0
        if((train_size+test_size)<=0.0 or (train_size+test_size)>=1.0):
            raise Exception("sum of values in tuples in splits should be <1.0")
        
        # splits in three sets - train, validation, test and returns
        train = data[:int(train_size*(len(data)))]
        val = data[int(train_size*(len(data))):int((train_size+test_size)*(len(data)))]
        test = data[int((train_size+test_size)*(len(data))):]
        return train, val, test