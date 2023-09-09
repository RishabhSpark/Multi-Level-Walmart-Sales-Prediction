from sklearn.metrics import mean_absolute_error as mae

def evaluation_metrics(actual, preds):
    '''
    Calculates MAE for the specified actual and predicted values
    Length of actual and preds should be equal
    
    Parameters
    ----------
    actual : Actual values
    preds : Predicted values
    
    Returns
    -------
    error_mae : float
                Mean average error
    '''
    # if actual does not equal to preds then raise error
    if len(actual)!=len(preds):
        raise ValueError("Length of actual does not equal to length of preds")
    
    error_mae = mae(actual, preds)
    return error_mae