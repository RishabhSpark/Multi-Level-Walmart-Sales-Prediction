import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarimax_model(X_train, X_test, y_train, y_test, X_val=None, y_val=None, **model_parameters):
    '''
    Creates a SARIMAX model and returns predictions for the train and test data
    
    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame
              Train dataset that will be used for training the model (features)
    y_train : pandas.core.frame.DataFrame
              Train dataset that will be used for training the model (target variable)
    X_test : pandas.core.frame.DataFrame
             Test dataset that will be used for evaluating the model (features)
    y_test : pandas.core.frame.DataFrame
             Test dataset that will be used for evaluating the model (target variable)
    X_val : NoneType or pandas.core.frame.DataFrame
            Default : None
            Validation dataset that will be used for evaluating the model (features)
    y_val : NoneType or pandas.core.frame.DataFrame
            Default : None
            Validation dataset that will be used for evaluating the model (target variable)
    model_parameters : kwargs
                       Keyword argument that can be used to provide parameters to the model
                       Can take statsmodels.tsa.statespace.sarimax.SARIMAX as parameters
            
    Returns
    -------
    train_preds : pandas.core.series.Series
                  Predictions made by the model on training dataset
    val_preds : pandas.core.series.Series
                Predictions made by the model on validation dataset (If X_val, y_val is not None)
    test_preds : pandas.core.series.Series
                 Predictions made by the model on testing dataset
    '''
    sarimax_model_fit = SARIMAX(y_train, exog=X_train, **model_parameters).fit()
    
    if X_val is None or y_val is None:
        train_preds = sarimax_model_fit.predict(start=1,end=len(X_train)-1, exog=X_train)
        test_preds = sarimax_model_fit.predict(start=len(X_train), end=len(X_train)+len(X_test)-1, exog=X_test)
        return train_preds, test_preds
    
    else:
        train_preds = sarimax_model_fit.predict(start=1,end=len(y_train)-1, exog=X_train)
        val_preds = sarimax_model_fit.predict(start=len(y_train), end=len(y_train)+len(y_val)-1, exog=X_val)
        test_preds = sarimax_model_fit.predict(start=len(y_train)+len(y_val), end=len(y_train)+len(y_val)+len(y_test)-1, exog=pd.concat([X_val,X_test]))
        test_preds = test_preds[-18:]
        return train_preds, val_preds, test_preds