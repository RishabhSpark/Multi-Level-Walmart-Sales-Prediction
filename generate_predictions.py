import prepare_data
import sarima_model
import ets_model
import sarimax_model
import evaluation
import plot
import grid_search

import pandas as pd

def generate_predictions_full(dataset, model_name, splits=0.8, perform_grid_search=False, **kwargs):
    '''
    Returns predictions for the specified model
    
    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
              Pandas univariate dataframe to be used for predictions
    
    model_name : {'SARIMA', 'ETS'} (Can accept in upper/lower/mixed case)
                 SARIMA : Creates a SARIMA model by calling the sarima_model function
                 ETS : Creates a triple Exponential Smoothing model by calling the ets_model function
    splits : float, tuple
             Default = 0.8; splitting data in two sets - train and test of size 0.8 and 0.2 respectively
             float : Splits in two sets - train and test
                     Only accepts value between range (0.0, 1.0)
             tuple(train_size, validation_size) : Splits in three sets - train, validation and test
                                                  train_size -> Train size
                                                  validation_size -> Validation size
                                                  Should only contain values between range (0.0, 1.0) and 
                                                  "train_size + test_size" should also be between (0.0, 1.0)
             Splits should be tuple if perform_grid_search is True
    perform_grid_search : boolean
                          Default = False; does not perform grid search
    **kwargs : kwargs
               Keyword arguments that can be used to provide parameters to the model or grid search
               If perform_grid_search=False & model_name='SARIMA' : kwargs =  statsmodels.tsa.statespace.sarimax.SARIMAX parameters
               If perform_grid_search=False & model_name='ETS' : kwargs =  statsmodels.tsa.holtwinters.ExponentialSmoothing parameters
               If perform_grid_search=True & model_name='SARIMA' : kwargs = grid_search_sarima function parameters
               If perform_grid_search=True & model_name='ETS' : kwargs = grid_search_ets function parameters
                                                      
    Prints
    ------
    Split size : float
                 If splits is float - Train and test splits 
                 If splits is a tuple - Train, validation, and test splits
    Grid Search Parameters and Range : Parameters provided by the user for grid search(if perform_grid_search==True)
    Best Parameters : Best parameters found by the grid search(if perform_grid_search==True)
    Model Parameters : Model parameters provided by the user (if perform_grid_search==False)
    Evaluation : Prints a dataframe containing "MAE" for train, val, test
    Prediction plot : Predictions on val and test data as a plot
    
    Returns
    -------
    val_preds : Predictions for val split (if splits is tuple)
    test_preds : Predictions for test split
    '''
    # check dataset type
    if not isinstance(dataset, pd.core.frame.DataFrame):
        raise TypeError("train is not a pandas dataframe")
        
    # check splits type
    if not isinstance(splits, (float, tuple)):
        raise TypeError("splits is not float or tuple")
    
    # prepares data and returns train, test if splits are float
    if isinstance(splits, float):
        train, test = prepare_data.prepare_data(dataset, splits)
    # prepares data and returns train, val, test if splits are tuple
    elif isinstance(splits, tuple):
        train, val, test = prepare_data.prepare_data(dataset, splits)
    
    # if model_name is not sarima or ets (uppercase/lowercase both works for user)
    model_name = model_name.upper()
    if (model_name not in ['SARIMA', 'ETS']):
        raise ValueError("model_name must be one of {'SARIMA', 'ETS'}")
    
    # perform_grid_search should be bool
    if not isinstance(perform_grid_search, bool):
        raise TypeError("perform_grid_search should be boolean")
        
    # if grid search then splits must be tuple for train/val/test
    if(perform_grid_search==True):
        if not isinstance(splits, tuple):
            raise ValueError("If perform_grid_search is True then splits should be a tuple")
            
    print("------------------Parameters------------------")
    # printing train/test split size
    if isinstance(splits, float):
        print("Train Split Size :", splits)
        print("Test Split Size :", round(1.0-splits, 3))

    # printing train/validation/test split size
    if isinstance(splits, tuple):
        print("Train Split Size :", splits[0])
        print("Validation Split Size :", splits[1])
        print("Test Split Size :", round(1.0-splits[1]-splits[0],3))
    
    # If grid search then performs grid search and returns the grid dataframe(as grid) and the best model parameters(as kwargs)
    if perform_grid_search==True:
        # printing user input range
        print("-----Parameters and Range for grid search-----")
        for key, value in kwargs.items():
            print(key, ":", value)
            
        _, kwargs = grid_search.grid_search(train, val, model_name, **kwargs)
        print("-------------Best Parameters Found-------------")
        for key, value in kwargs.items():
            print(key, ":", value)
        
        if model_name=='SARIMA':
            train_preds, val_preds, test_preds = sarima_model.sarima_model(train, test, val=val, **kwargs)
            
            train_error_mae = evaluation.evaluation_metrics(train[1:], train_preds)
            val_error_mae = evaluation.evaluation_metrics(val, val_preds)
            test_error_mae = evaluation.evaluation_metrics(test, test_preds)
        
        elif model_name=='ETS':
            train_preds, val_preds, test_preds = ets_model.ets_model(train, test, val=val, **kwargs)

            train_error_mae = evaluation.evaluation_metrics(train, train_preds)
            val_error_mae = evaluation.evaluation_metrics(val, val_preds)
            test_error_mae = evaluation.evaluation_metrics(test, test_preds)
    
        # mae
        error_df = pd.DataFrame([[round(train_error_mae, 4)],
                                 [round(val_error_mae, 4)],
                                 [round(test_error_mae, 4)]],
                                columns = ['MAE'],
                                index = ['Train', 'Val', 'Test'])
    
        plot.val_prediction_plot(model_name, train, val, test, val_preds, test_preds)
        print("------------------Evaluation------------------")
        print(error_df)
    
        return val_preds, test_preds
    
    # If no grid search
    else:
        print("------------------Parameters------------------")
        for key, value in kwargs.items():
            print(key, ":", value)
   
        if isinstance(splits, float):
            # for sarima model perform
            if model_name=='SARIMA':
                train_preds, test_preds = sarima_model.sarima_model(train, test, **kwargs)

                train_error_mae = evaluation.evaluation_metrics(train[1:], train_preds)
                test_error_mae = evaluation.evaluation_metrics(test, test_preds)

            # for ets perform
            elif model_name=='ETS':
                train_preds, test_preds = ets_model.ets_model(train, test, **kwargs)

                train_error_mae = evaluation.evaluation_metrics(train, train_preds)
                test_error_mae = evaluation.evaluation_metrics(test, test_preds)

            # mae
            error_df = pd.DataFrame([[round(train_error_mae, 4)],
                                     [round(test_error_mae, 4)]], 
                                    columns = ['MAE'], 
                                    index = ['Train', 'Test'])
            
            plot.prediction_plot(model_name, train, test, test_preds)
            print("------------------Evaluation------------------")
            print(error_df)

            return test_preds
        else:
            if model_name=='SARIMA':
                train_preds, val_preds, test_preds = sarima_model.sarima_model(train, test, val=val, **kwargs)
                train_error_mae = evaluation.evaluation_metrics(train[1:], train_preds)
                val_error_mae = evaluation.evaluation_metrics(val, val_preds)
                test_error_mae = evaluation.evaluation_metrics(test, test_preds)
        
            elif model_name=='ETS':
                train_preds, val_preds, test_preds = ets_model.ets_model(train, test, val=val, **kwargs)

                train_error_mae = evaluation.evaluation_metrics(train, train_preds)
                val_error_mae = evaluation.evaluation_metrics(val, val_preds)
                test_error_mae = evaluation.evaluation_metrics(test, test_preds)

            # mae
            error_df = pd.DataFrame([[round(train_error_mae, 4)],
                                     [round(val_error_mae, 4)],
                                     [round(test_error_mae, 4)]],
                                    columns = ['MAE'],
                                    index = ['Train', 'Val', 'Test'])
            
            plot.val_prediction_plot(model_name, train, val, test, val_preds, test_preds)
            print("------------------Evaluation------------------")
            print(error_df)

            return val_preds, test_preds
        
        
        
def sarimax_full(dataset, splits=0.8, perform_grid_search=False, **kwargs):
    '''
    Returns predictions for the specified model
    
    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
              Pandas univariate dataframe to be used for predictions
    splits : float, tuple
             Default = 0.8; splitting data in two sets - train and test of size 0.8 and 0.2 respectively
             float : Splits in two sets - train and test
                     Only accepts value between range (0.0, 1.0)
             tuple(train_size, validation_size) : Splits in three sets - train, validation and test
                                                  train_size -> Train size
                                                  validation_size -> Validation size
                                                  Should only contain values between range (0.0, 1.0) and 
                                                  "train_size + test_size" should also be between (0.0, 1.0)
             Splits should be tuple if perform_grid_search is True
    perform_grid_search : boolean
                          Default = False; does not perform grid search
    **kwargs : kwargs
               Keyword arguments that can be used to provide parameters to the model or grid search
               If perform_grid_search=False : kwargs =  statsmodels.tsa.statespace.sarimax.SARIMAX parameters
               If perform_grid_search=True : kwargs = grid_search_sarimax function parameters
                                                      
    Prints
    ------
    Split size : float
                 If splits is float - Train and test splits 
                 If splits is a tuple - Train, validation, and test splits
    Grid Search Parameters and Range : Parameters provided by the user for grid search(if perform_grid_search==True)
    Best Parameters : Best parameters found by the grid search(if perform_grid_search==True)
    Model Parameters : Model parameters provided by the user (if perform_grid_search==False)
    Evaluation : Prints a dataframe containing "MAE" for train, val, test
    Prediction plot : Predictions on val and test data as a plot
    
    Returns
    -------
    val_preds : Predictions for val split (if splits is tuple)
    test_preds : Predictions for test split
    '''
    # perform_grid_search should be bool
    if not isinstance(perform_grid_search, bool):
        raise TypeError("perform_grid_search should be boolean")
        
    # if grid search then splits must be tuple for train/val/test
    if(perform_grid_search==True):
        if not isinstance(splits, tuple):
            raise ValueError("If perform_grid_search is True then splits should be a tuple")
            
    # print parameters
    print("------------------Parameters------------------")
    # printing train/test split size
    if isinstance(splits, float):
        print("Train Split Size :", splits)
        print("Test Split Size :", round(1.0-splits, 3))

    # printing train/validation/test split size
    if isinstance(splits, tuple):
        print("Train Split Size :", splits[0])
        print("Validation Split Size :", splits[1])
        print("Test Split Size :", round(1.0-splits[1]-splits[0],3))
            
    # prepares data and returns train, test if splits are float
    if isinstance(splits, float):
        train, test = prepare_data.prepare_data(dataset, splits)
        train_Y = train['Weekly_Sales']
        train_X = train.drop(columns='Weekly_Sales')

        test_Y = test['Weekly_Sales']
        test_X = test.drop(columns='Weekly_Sales')
    # prepares data and returns train, val, test if splits are tuple
    elif isinstance(splits, tuple):
        train, val, test = prepare_data.prepare_data(dataset, splits)
        train_Y = train['Weekly_Sales']
        train_X = train.drop(columns='Weekly_Sales')
        
        val_Y = val['Weekly_Sales']
        val_X = val.drop(columns='Weekly_Sales')
        
        test_Y = test['Weekly_Sales']
        test_X = test.drop(columns='Weekly_Sales')
    
    if perform_grid_search==True:
        # printing user input range
        print("-----Parameters and Range for grid search-----")
        for key, value in kwargs.items():
            print(key, ":", value)

        _, kwargs = grid_search.grid_search_sarimax(train_X, val_X, train_Y, val_Y, **kwargs)
        print("-------------Best Parameters Found-------------")
        for key, value in kwargs.items():
            print(key, ":", value)
            
        train_preds, val_preds, test_preds = sarimax_model.sarimax_model(train_X, test_X, train_Y, test_Y, X_val=val_X, y_val=val_Y, **kwargs)
        train_error_mae = evaluation.evaluation_metrics(train_Y[1:], train_preds)
        val_error_mae = evaluation.evaluation_metrics(val_Y, val_preds)
        test_error_mae = evaluation.evaluation_metrics(test_Y, test_preds)
    
        # mae
        error_df = pd.DataFrame([[round(train_error_mae, 4)],
                                 [round(val_error_mae, 4)],
                                 [round(test_error_mae, 4)]],
                                columns = ['MAE'],
                                index = ['Train', 'Val', 'Test'])
    
        plot.val_prediction_plot("SARIMAX", train_Y, val_Y, test_Y, val_preds, test_preds)
        print("------------------Evaluation------------------")
        print(error_df)
    
        return val_preds, test_preds
    
    else:
        print("------------------Parameters------------------")
        for key, value in kwargs.items():
            print(key, ":", value)
        
        if isinstance(splits, float):         
            train_preds, test_preds = sarimax_model.sarimax_model(train_X, test_X, train_Y, test_Y, **kwargs)
            
            train_error_mae = evaluation.evaluation_metrics(train_Y[1:], train_preds)
            test_error_mae = evaluation.evaluation_metrics(test_Y, test_preds)
            # mae
            error_df = pd.DataFrame([[round(train_error_mae, 4)],
                                     [round(test_error_mae, 4)]], 
                                    columns = ['MAE'], 
                                    index = ['Train', 'Test'])
            
            plot.prediction_plot("SARIMAX", train_Y, test_Y, test_preds)
            print("------------------Evaluation------------------")
            print(error_df)

            return test_preds
        
        else:
            train_preds, val_preds, test_preds = sarimax_model.sarimax_model(train_X, test_X, train_Y, test_Y, val_X, val_Y, **kwargs)
            train_error_mae = evaluation.evaluation_metrics(train_Y[1:], train_preds)
            val_error_mae = evaluation.evaluation_metrics(val_Y, val_preds)
            test_error_mae = evaluation.evaluation_metrics(test_Y, test_preds)

            # mae
            error_df = pd.DataFrame([[round(train_error_mae, 4)],
                                     [round(val_error_mae, 4)],
                                     [round(test_error_mae, 4)]],
                                    columns = ['MAE'],
                                    index = ['Train', 'Val', 'Test'])
            
            plot.val_prediction_plot("SARIMAX", train_Y, val_Y, test_Y, val_preds, test_preds)
            print("------------------Evaluation------------------")
            print(error_df)

            return val_preds, test_preds
        
        
        
def generate_predictions(dataset, model_name, splits=0.8, perform_grid_search=False, **kwargs):
    '''
    Returns predictions for the specified model by calling either generate_predictions_full or sarimax_full functions
    
    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame
              Pandas univariate dataframe to be used for predictions
    
    model_name : {'SARIMA', 'ETS', 'SARIMAX'} (Can accept in upper/lower/mixed case)
                 SARIMA : Creates a SARIMA model
                 ETS : Creates a triple Exponential Smoothing model
                 SARIMAX : Creates a SARIMAX model
    splits : float, tuple
             Default = 0.8; splitting data in two sets - train and test of size 0.8 and 0.2 respectively
             float : Splits in two sets - train and test
                     Only accepts value between range (0.0, 1.0)
             tuple(train_size, validation_size) : Splits in three sets - train, validation and test
                                                  train_size -> Train size
                                                  validation_size -> Validation size
                                                  Should only contain values between range (0.0, 1.0) and 
                                                  "train_size + test_size" should also be between (0.0, 1.0)
             Splits should be tuple if perform_grid_search is True
    perform_grid_search : boolean
                          Default = False; does not perform grid search
    **kwargs : kwargs
               Keyword arguments that can be used to provide parameters to the model or grid search of the specified model
    
    Returns
    -------
    val_preds : Predictions for val split (if splits is tuple)
    test_preds : Predictions for test split
    '''
    model_name=model_name.upper()
    if (model_name not in ['SARIMA', 'ETS', 'SARIMAX']):
        raise ValueError("model_name must be one of {'SARIMA', 'ETS', 'SARIMAX'}")
        
    if model_name in ['SARIMA', 'ETS']:
        if perform_grid_search==False:
            test_preds = generate_predictions_full(dataset, model_name, splits, perform_grid_search, **kwargs)
            return test_preds
        
        elif perform_grid_search==True:
            val_preds, test_preds = generate_predictions_full(dataset, model_name, splits, perform_grid_search, **kwargs)
            return val_preds, test_preds
        
    elif model_name=='SARIMAX':
        if perform_grid_search==False:
            test_preds = sarimax_full(dataset, splits, perform_grid_search, **kwargs)
            return test_preds
        elif perform_grid_search==True:
            val_preds, test_preds = sarimax_full(dataset, splits, perform_grid_search, **kwargs)
            return val_preds, test_preds