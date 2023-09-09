import matplotlib.pyplot as plt

def prediction_plot(plot_title, plot_train, plot_test, plot_preds):
    '''
    Creates a matplotlib plot that displays train, test, and predicted values over time
    
    Parameters
    ----------
    plot_title : Title of the plot
    plot_train : Actual data values over time
    plot_test : Test values over time
    plot_preds : Predicted values over time
    
    Returns
    -------
    Matplotlib line chart
    '''
    plt.figure(figsize=(20,6))
    plt.title(plot_title)
    plt.plot(plot_train, label='Train')
    plt.plot(plot_test, label='Test')
    plt.plot(plot_preds, label='Prediction')
    plt.legend()
    plt.show()
    
    
def val_prediction_plot(plot_title, plot_train, plot_val, plot_test, plot_val_preds, plot_test_preds):
    '''
    Creates a matplotlib plot that displays train, test, and predicted values over time
    
    Parameters
    ----------
    plot_title : Title of the plot
    plot_train : Actual data values over time
    plot_val : Validation values over time
    plot_test : Test values over time
    plot_val_preds : Predicted values on validation data over time
    plot_test_preds : Predicted_values on test data over time
    
    Returns
    -------
    Matplotlib line chart
    '''
    plt.figure(figsize=(20,6))
    plt.title(plot_title)
    plt.plot(plot_train, label='Train')
    plt.plot(plot_val, label='Validation')
    plt.plot(plot_test, label='Test')
    plt.plot(plot_val_preds, label='Validation Prediction')
    plt.plot(plot_test_preds, label='Test Predictions')
    plt.legend()
    plt.show()