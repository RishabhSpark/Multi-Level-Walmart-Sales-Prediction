import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_model(train, test, val=None, **model_parameters):
    '''
    Creates a SARIMA model and returns predictions for the train and test data
    
    Parameters
    ----------
    train : pandas.core.frame.DataFrame
            Train dataset that will be used for training the model
    test : pandas.core.frame.DataFrame
            Test dataset that will be used for evaluating the model
    val : NoneType or pandas.core.frame.DataFrame
          Default : None
          Validation dataset that will be used for evaluating the model
    model_parameters : kwargs
                       Keyword argument that can be used to provide parameters to the model
                       Can take statsmodels.tsa.statespace.sarimax.SARIMAX as parameters
            
    Returns
    -------
    train_preds : pandas.core.series.Series
                  Predictions made by the model on training dataset
    val_preds : pandas.core.series.Series
                Predictions made by the model on validation dataset (If val is not None)
    test_preds : pandas.core.series.Series
                 Predictions made by the model on testing dataset
    '''

    # check if train and test are pandas dataframe
    if not isinstance(train, pd.core.frame.DataFrame):
        raise TypeError("train is not a pandas dataframe")
    
    if not isinstance(test, pd.core.frame.DataFrame):
        raise TypeError("test is not a pandas dataframe")
    
    # If val is not pandas dataframe or not NoneType
    if not isinstance(val, pd.core.frame.DataFrame):
        if val is not None:
            raise TypeError("val is not a pandas dataframe")
    
    # model creation + fit
    sarima_model = SARIMAX(train,  **model_parameters)
    sarima_model_fit = sarima_model.fit()

    if val is None:
        # predictions for train and test
        train_preds = sarima_model_fit.predict(start=1,end=len(train)-1)
        test_preds = sarima_model_fit.predict(start=len(train), end=len(train)+len(test)-1)
        return train_preds, test_preds
    
    else:
        # predictions for train, val, test
        train_preds = sarima_model_fit.predict(start=1,end=len(train)-1)
        val_preds = sarima_model_fit.predict(start=len(train), end=len(train)+len(val)-1)
        test_preds = sarima_model_fit.predict(start=len(train)+len(val), end=len(train)+len(val)+len(test)-1)
        return train_preds, val_preds, test_preds