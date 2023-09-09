import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def ets_model(train, test, val=None, **model_parameters):
    '''
    Creates a Triple Exponential Smoothing Model by calling statsmodels.tsa.holtwinters.ExponentialSmoothing and returns predictions for the train, val and test data
    
    Parameters
    ----------
    train : pandas.core.frame.DataFrame
            Train dataset that will be used for training the model
    test : pandas.core.frame.DataFrame
           Test dataset that will be used for evaluating the model
    val : NoneType or pandas.core.frame.DataFrame
          Default : None
          Validation dataset that will be used for evaluating the model
    model_parameters: kwargs
                      Keyword argument that can be used to provide parameters to the model
                      Can take statsmodels.tsa.holtwinters.ExponentialSmoothing parameters and
                      smoothing_level, smoothing_trend, smoothing_seasonal parameters of 
                      statsmodels.tsa.holtwinters.ExponentialSmoothing.fit
                      default smoothing_level = 0.5
                      default smoothing_trend = 0.5
                      default smoothing seasonal = 0.5
            
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

    # default fit parameters
    smoothing_level=0.5
    smoothing_trend=0.5
    smoothing_seasonal=0.5
    
    # If fit parameters are specified by user with model_parameters then removing those from model_parameters 
    #and updating default fit parameters to whatever is specified as input
    if('smoothing_level' in model_parameters):
        smoothing_level = model_parameters['smoothing_level']
        model_parameters.pop('smoothing_level')
        
    if('smoothing_trend' in model_parameters):
        smoothing_trend = model_parameters['smoothing_trend']
        model_parameters.pop('smoothing_trend')
        
    if('smoothing_seasonal' in model_parameters):
        smoothing_seasonal = model_parameters['smoothing_seasonal']
        model_parameters.pop('smoothing_seasonal')

    # model creation + fit
    EXPmodel = ExponentialSmoothing(train, **model_parameters).fit(smoothing_level=smoothing_level,
                                                                   smoothing_trend=smoothing_trend,
                                                                   smoothing_seasonal=smoothing_seasonal)

    if val is None:
        # predictions for train and test
        train_preds = EXPmodel.predict(start=0,end=len(train)-1)
        test_preds = EXPmodel.predict(start=len(train), end=len(train)+len(test)-1)
        return train_preds, test_preds
    
    else:
        # predictions for train, val, test
        train_preds = EXPmodel.predict(start=0,end=len(train)-1)
        val_preds = EXPmodel.predict(start=len(train), end=len(train)+len(val)-1)
        test_preds = EXPmodel.predict(start=len(train)+len(val), end=len(train)+len(val)+len(test)-1)
        return train_preds, val_preds, test_preds