""" 

author: @cesarasa

Code for the measurements of the model.

My goal is to measure the NRMSE, PCC, Bias, and NMAE.

NOTE: PCC is pearson's correlation, thus we can use a function from numpy to calculate it.

"""

import numpy as np
# Import rsquare
from sklearn.metrics import r2_score

# Import Linear Regression:
from sklearn.linear_model import LinearRegression
def nRMSE(y_observed : np.array,
          y_predicted :np.array) -> tuple:
    """
    This code is to recreate the NRMSE function presented by Omid Bazgir
    and Ruibo Zhang in the REFINED paper. 
    
    Source: https://www.nature.com/articles/s41467-020-18197-y
    Code Source: https://github.com/omidbazgirTTU/REFINED/blob/master/Toolbox.py

    Args:
        y_observed (np.array) : array with the target y.
        y_predicted (np.array): array with the predictions y_hat.

    Returns:
        tuple with the NRMSE, R2, MSE and RMSE. In that order.
    """

    # Assert if the inputs are numpy arrays:
    assert isinstance(y_observed, np.ndarray), 'y_observed should be a numpy array'
    assert isinstance(y_predicted, np.ndarray), 'y_predicted should be a numpy array'

    # Reshape
    y_observed  = y_observed.reshape(len(y_observed), 1)
    y_predicted = y_predicted.reshape(len(y_predicted), 1)

    # Assert if the inputs have the same shape:
    assert y_observed.shape == y_predicted.shape, 'y_observed and y_predicted should have the same shape'


    # Calculate the distance (l2) of the two vectors
    dist = np.sum((y_predicted - y_observed)**2)
    mean_observed = np.mean(y_observed)
    dist_to_mean = np.sum((mean_observed - y_observed)**2)
    nrmse = np.sqrt(dist/dist_to_mean) 
    mse = np.mean((y_predicted - y_observed)**2)
    rmse = np.sqrt(mse)

    # Computing r2 without the function:
    r2 = 1 - nrmse**2
    return nrmse, r2, mse, rmse

def NMAE(y_observed : np.array,
         y_predicted :np.array) -> tuple:
    """
    This code is to recreate the nMAE function presented by Omid Bazgir
    and Ruibo Zhang in the REFINED paper. 
    
    Source: https://www.nature.com/articles/s41467-020-18197-y
    Code Source: https://github.com/omidbazgirTTU/REFINED/blob/master/Toolbox.py
    
    Args:
        y_observed  (np.array) : array with the target y.
        y_predicted (np.array) : array with the predictions y_hat

    Returns:
        Tuple with the MAE and nMAE.
    """
    # Check if they are numpy arrays
    assert isinstance(y_observed, np.ndarray), 'y_observed should be a numpy array'
    assert isinstance(y_predicted, np.ndarray), 'y_predicted should be a numpy array'
    
    # Reshape
    y_observed  = y_observed.reshape(len(y_observed), 1)
    y_predicted = y_predicted.reshape(len(y_predicted), 1)
    
    # Check if they have the same shape
    assert y_observed.shape == y_predicted.shape, 'y_observed and y_predicted should have the same shape'
    
    # Compute the mean of the observations:
    observed_mean = np.mean(y_observed)
    
    # Compute the distance between the two vectors:
    dist = np.sum(np.abs(y_observed - y_predicted))
    dist_to_mean = np.sum(np.abs(y_observed - observed_mean))
    nmae = dist/dist_to_mean
    
    # The following lines are how Omid/Ruibo coded it, it gives the same result.
    # NOTE: Check if it is giving the same result out of chance or if it makes sense.
    # nom = np.abs(y_observed - y_predicted)
    # den = np.abs(y_observed - observed_mean)
    # nmae2 = np.mean(nom)/np.mean(den)
    # Compute MAE
    mae = np.mean(np.abs(y_observed - y_predicted))
    
    return nmae, mae

def Bias(y_observed  : np.array,
         y_predicted :np.array) -> tuple:
    
    # Assert numpy:
    assert isinstance(y_observed, np.ndarray),  'y_observed should be a numpy array'
    assert isinstance(y_predicted, np.ndarray), 'y_predicted should be a numpy array'
    
    # Reshape
    y_observed  = y_observed.reshape(len(y_observed), 1)
    y_predicted = y_predicted.reshape(len(y_predicted), 1)
    
    # Check if they have the same shape
    assert y_observed.shape == y_predicted.shape, 'y_observed and y_predicted should have the same shape'
    
    # Now we need to compute the error vector:
    error = y_observed - y_predicted
    
    # We need to get the angle between the error and the predictions:
    cos_angle = y_observed.T @ error / (np.linalg.norm(y_observed) * np.linalg.norm(error))
    angle = np.arccos(cos_angle)
    
    # Bias:
    bias = np.tan(angle)
    
    # According to Ruibo:
    reg = LinearRegression().fit(y_observed, error)
    bias2 = reg.coef_[0]  # WHY?
    return float(bias[0]), float(bias2), float(angle)
    



def main_test():
    # TEST NRMSE FUNCTION:
    y_observed = np.load('logs/labels_array.npy')
    y_predicted = np.load('logs/predictions_array.npy')
    print(y_predicted.shape)
    print(y_observed.shape)
    #TEST
    print('Checking NRMSE')
    print(NRMSE(y_observed, y_predicted))
    # Not using any function
    print(np.sqrt(1-r2_score(y_observed, y_predicted)))
    # NOTE: Omid is right.
    print('Checking NMAE')
    print(NMAE(y_observed, y_predicted))
    # NOTE: Omid might be right. 
    print('Checking Bias')
    print(Bias(y_observed, y_predicted)) # This is not getting the same result, debug. 
    # Ask Dr. Pal or check with Ayda. 
    return None


if __name__ == '__main__':
    main_test()
