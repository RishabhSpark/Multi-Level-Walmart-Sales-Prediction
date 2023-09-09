import sarima_model
import ets_model
import sarimax_model
import evaluation

import numpy as np
import pandas as pd
import itertools


def grid_search_sarima(train, val,
                       p_range=range(0,2), d_range=range(0,2), q_range=range(0,2),
                       P_range=range(0,2), D_range=range(0,2), Q_range=range(0,2), m=[52]):
    '''
    Performs a grid search on statsmodels.tsa.statespace.sarimax.SARIMAX
    
    Parameters
    ----------
    train : Pandas.core.frame.DataFrame
            Train set
    val : Pandas.core.frame.DataFrame
          Validation set
    p_range : range
              Default : range(0,2)
              Range of trend autoregression order
    d_range : range
              Default : range(0,2)
              Range of trend difference order
    q_range : range
              Default : range(0,2)
              Range of trend moving average order
    P_range : range
              Default : range(0,2)
              Range of seasonal autoregression order
    D_range : range
              Default : range(0,2)
              Range of seasonal difference order
    Q_range : range
              Default : range(0,2)
              Range of seasonal moving average order
    m : list
        Default : [52]
        The number of time steps for a single seasonal period
    
    Returns
    -------
    grid : pandas.core.frame.DataFrame
           Returns a dataframe with columns=['order', 'seasonal_order', 'train_error', 'val_error', 'test_error']
    best_params : dict({'order':order_parameters, 'seasonal_order':seasonal_order_parameters})
                  Returns a key-value pair of best performing model
    '''
    # check if train and test are pandas dataframe
    if not isinstance(train, pd.core.frame.DataFrame):
        raise TypeError("train is not a pandas dataframe")
  
    if not isinstance(val, pd.core.frame.DataFrame):
        raise TypeError("val is not a pandas dataframe")
        
    # check if the range variables are python range
    range_inputs = [p_range, d_range, q_range, P_range, D_range, Q_range]
    for i in range_inputs:
        if not isinstance(i, range):
            raise TypeError("Range inputs are not of type range")
    
    # check if m is a python list
    if not isinstance(m, list):
        raise TypeError("m is not a list")

    params_combinations = list(itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range, m))
    
    # defining return variables
    best_error=float("inf")
    best_params=None
    
    # defining output of each value as pandas dataframe
    grid = pd.DataFrame(columns=['order', 'seasonal_order', 'train_error', 'val_error'])
    
    # unpacking itertools product params_combinations
    for p, d, q, P, D, Q, m in params_combinations:
        order = (p, d, q)
        seasonal_order = (P, D, Q, m)
        
        # try except for handling exceptions
        try:
            # create the model and returns predictions for train and test
            train_preds, val_preds = sarima_model.sarima_model(train, val, order=order, seasonal_order=seasonal_order)

            # if preds are nan then handle
            if(np.isnan(np.min(train_preds)) or np.isnan(np.min(val_preds))):
                grid.loc[len(grid)] = [order, seasonal_order, 'nan', 'nan']
            
            # if preds are non-nan
            # evaluates the model and assigns best error and best params if the test/val error is least
            else:
                train_error = evaluation.evaluation_metrics(train[1:], train_preds)
                val_error = evaluation.evaluation_metrics(val, val_preds)

                grid.loc[len(grid)] = [order, seasonal_order, train_error, val_error]

            if(val_error<best_error):
                best_error=val_error
                best_params = {'order':order, 'seasonal_order':seasonal_order}
    
        except Exception as e:
            continue
    
    # returning best parameters
    return grid, best_params


def grid_search_ets(train, val,
                    trend=['add', 'mul'], seasonal=['add','mul'], seasonal_periods=[52], 
                    smoothing_values=np.arange(0.0,0.3,0.1)):
    '''
    Performs a grid search on statsmodels.tsa.holtwinters.ExponentialSmoothing
    
    Parameters
    ----------
    train : Pandas.core.frame.DataFrame
            Train set
    val : Pandas.core.frame.DataFrame
          Validation set
    trend : list
            Default : ['add', 'mul']
            Trend component of statsmodels.tsa.holtwinters.ExponentialSmoothing
            Can have only 'add' and/or 'mul' as values inside list
    seasonal : list
               Default : ['add', 'mul']
               Seasonal component of statsmodels.tsa.holtwinters.ExponentialSmoothing
               Can have only 'add' and/or 'mul' as values inside list
    seasonal_periods : list
                       Default : [52]
                       Seasonal_periods component of statsmodels.tsa.holtwinters.ExponentialSmoothing
    smoothing_values : numpy.ndarray
                       Default : np.arange(0.0,0.3,0.1)
                       Set of values that can be used for for smoothing_level, smoothing_trend, smoothing_seasonal values of
                       statsmodels.tsa.holtwinters.ExponentialSmoothing.fit
    
    Returns
    -------
    grid : pandas.core.frame.DataFrame
           Returns a dataframe with columns=['trend', 'seasonal', 'seasonal_periods', 
                                             'smoothing_level', 'smoothing_trend', 'smoothing_seasonal',
                                             'train_error', 'val_error', 'test_error']
    best_params : dict({'trend':trend,'seasonal':seasonal,'seasonal_periods':seasonal_periods,
                        'smoothing_level':smoothing_level, 'smoothing_trend':smoothing_trend,'smoothing_seasonal':smoothing_seasonal})
                  Returns a key-value pair of the best performing model
              
    '''
    # check if train and test are pandas dataframe
    if not isinstance(train, pd.core.frame.DataFrame):
        raise TypeError("train is not a pandas dataframe")
     
    if not isinstance(val, pd.core.frame.DataFrame):
        raise TypeError("val is not a pandas dataframe")
    
    
    list_inputs = [trend, seasonal]    
    for col in list_inputs:
        # check if the list variables are python range
        if not isinstance(col, list):
            raise TypeError("Trend and/or seasonal is not of type list")
            
        # check len of list_inputs as it can only have 1 or 2 values
        if(len(col)>2 or len(col)<1):
            raise ValueError("Trend and seasonal can only have 1 or 2 values")

        for i in col:
            if(i!='add' and i!='mul'):
                raise ValueError("Trend and seasonal can only have 'add' and/or 'mul' as values in the list")
        
    # check if smoothing_values is of correct type
    if not isinstance(smoothing_values, np.ndarray):
        raise TypeError("Smoothing range is not of type ")

    smoothing_level = smoothing_values
    smoothing_trend = smoothing_values
    smoothing_seasonal = smoothing_values
    
    params_combinations = list(itertools.product(trend, seasonal, seasonal_periods, 
                                                 smoothing_level, smoothing_trend, smoothing_seasonal))
    
    # defining return variables
    best_error=float("inf")
    best_params=None
    
    # defining output of each value as pandas dataframe
    grid = pd.DataFrame(columns=['trend', 'seasonal', 'seasonal_periods', 
                                 'smoothing_level', 'smoothing_trend', 'smoothing_seasonal',
                                 'train_error', 'val_error'])
    
    # unpacking itertools product params_combinations
    for trend, seasonal, seasonal_periods, alpha, beta, gamma in params_combinations:
        
        # try except for handling exceptions
        try:
            # create the model and returns predictions for train and test
            train_preds, val_preds = ets_model(train, val,
                                               trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods,
                                               smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)

            # if preds are nan then handle
            if(np.isnan(np.min(train_preds)) or np.isnan(np.min(val_preds))):
                grid.loc[len(grid)] = [trend, seasonal, seasonal_periods,
                                       alpha, beta, gamma, 
                                       'nan', 'nan']
            
            # if preds are non-nan
            # evaluates the model and assigns best error and best params if the test/val error is least
            else:
                train_error = evaluation.evaluation_metrics(train, train_preds)
                val_error = evaluation.evaluation_metrics(val, val_preds)

                grid.loc[len(grid)] = [trend, seasonal, seasonal_periods,
                                       alpha, beta, gamma,
                                       train_error, val_error]

            if(val_error<best_error):
                best_error=val_error
                best_params = {'trend':trend,
                               'seasonal':seasonal,
                               'seasonal_periods':seasonal_periods,
                               'smoothing_level':alpha,
                               'smoothing_trend':beta,
                               'smoothing_seasonal':gamma}
    
        except Exception as e:
            continue
    
    # printing the grid dataframe and returning best parameters
    return grid, best_params


def grid_search_sarimax(X_train, X_val, y_train, y_val,
                        p_range=range(0,2), d_range=range(0,2), q_range=range(0,2),
                        P_range=range(0,2), D_range=range(0,2), Q_range=range(0,2), m=[52]):
    '''
    Performs a grid search on statsmodels.tsa.statespace.sarimax.SARIMAX
    
    Parameters
    ----------
    X_train : Pandas.core.frame.DataFrame
              Train set exogenous variables
    X_val : Pandas.core.frame.DataFrame
            Validation set exogenous variables
    y_train : Pandas.core.frame.DataFrame
              Train set output variable
    y_val : Pandas.core.frame.DataFrame
            Validation set output variable
    p_range : range
              Default : range(0,2)
              Range of trend autoregression order
    d_range : range
              Default : range(0,2)
              Range of trend difference order
    q_range : range
              Default : range(0,2)
              Range of trend moving average order
    P_range : range
              Default : range(0,2)
              Range of seasonal autoregression order
    D_range : range
              Default : range(0,2)
              Range of seasonal difference order
    Q_range : range
              Default : range(0,2)
              Range of seasonal moving average order
    m : list
        Default : [52]
        The number of time steps for a single seasonal period
    
    Returns
    -------
    grid : pandas.core.frame.DataFrame
           Returns a dataframe with columns=['order', 'seasonal_order', 'train_error', 'val_error', 'test_error']
    best_params : dict({'order':order_parameters, 'seasonal_order':seasonal_order_parameters})
                  Returns a key-value pair of best performing model
    '''
        
    # check if the range variables are python range
    range_inputs = [p_range, d_range, q_range, P_range, D_range, Q_range]
    for i in range_inputs:
        if not isinstance(i, range):
            raise TypeError("Range inputs are not of type range")
    
    # check if m is a python list
    if not isinstance(m, list):
        raise TypeError("m is not a list")

    params_combinations = list(itertools.product(p_range, d_range, q_range, P_range, D_range, Q_range, m))
    
    # defining return variables
    best_error=float("inf")
    best_params=None
    
    # defining output of each value as pandas dataframe
    grid = pd.DataFrame(columns=['order', 'seasonal_order', 'train_error', 'val_error'])
    
    # unpacking itertools product params_combinations
    for p, d, q, P, D, Q, m in params_combinations:
        order = (p, d, q)
        seasonal_order = (P, D, Q, m)
        
        # try except for handling exceptions
        try:
            # create the model and returns predictions for train and test
            train_preds, val_preds = sarimax_model.sarimax_model(X_train, X_val, y_train, y_val, order=order, seasonal_order=seasonal_order)

            # if preds are nan then handle
            if(np.isnan(np.min(train_preds)) or np.isnan(np.min(val_preds))):
                grid.loc[len(grid)] = [order, seasonal_order, 'nan', 'nan']
            
            # if preds are non-nan
            # evaluates the model and assigns best error and best params if the test/val error is least
            else:
                train_error = evaluation.evaluation_metrics(y_train[1:], train_preds)
                val_error = evaluation.evaluation_metrics(y_val, val_preds)

                grid.loc[len(grid)] = [order, seasonal_order, train_error, val_error]

            if(val_error<best_error):
                best_error=val_error
                best_params = {'order':order, 'seasonal_order':seasonal_order}
    
        except Exception as e:
            continue
    
    # returning best parameters
    return grid, best_params


def grid_search(train, val, model_name, **parameters):
    '''
    Performs a grid search on the given model
    
    Parameters
    ----------
    train : Pandas.core.frame.DataFrame
            Train set
    val : Pandas.core.frame.DataFrame
          Validation set
    test : Pandas.core.frame.DataFrame
           Test set
    model_name : {'SARIMA', 'ETS'} (Can accept this in upper/lower/mixed case)
                 SARIMA : Creates a SARIMA model
                 ETS : Creates a triple Exponential Smoothing model
    **parameters : kwargs
                   Parameters that can be passed in grid search
                   
    Returns
    -------
    grid : pandas.core.frame.DataFrame
           Returns a dataframe of all the results of exhaustive search
           
           If model_name = 'SARIMA' : Returns a dataframe with 
                                      columns=['order', 'seasonal_order', 'train_error', 'test_error']
           If model_name = 'ETS' : Returns a dataframe with 
                                   columns=['trend', 'seasonal', 'seasonal_periods',
                                            'smoothing_level', 'smoothing_trend', 'smoothing_seasonal',
                                            'train_error', 'test_error']
                                                         
    best_params : Dictionary
                  Returns a key-value pair of the best performing model   
                  If model_name = 'SARIMA' : dict({'order':order_parameters, 
                                                   'seasonal_order':seasonal_order_parameters})
                  If model_name = 'ETS' : dict({'trend':trend,
                                                'seasonal':seasonal,
                                                'seasonal_periods':seasonal_periods,
                                                'smoothing_level':smoothing_level, 
                                                'smoothing_trend':smoothing_trend,
                                                'smoothing_seasonal':smoothing_seasonal})
    '''
    # if model_name is not sarima or ets (uppercase/lowercase both works for user)
    model_name = model_name.upper()
    if (model_name not in ['SARIMA', 'ETS']):
        raise ValueError("model_name must be one of {'SARIMA', 'ETS'}")
    
    model_name = model_name.upper()
    # Performs SARIMA grid search
    if(model_name=='SARIMA'):
        grid, best_parameters = grid_search_sarima(train, val, **parameters)

    # Performs ETS grid search
    elif(model_name=='ETS'):
        grid, best_parameters = grid_search_ets(train, val, **parameters)
    
    return grid, best_parameters