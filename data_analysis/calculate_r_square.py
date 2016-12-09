import numpy as np

def compute_r_squared(data, predictions):
    # Write a function that, given two input numpy arrays, 'data', and 'predictions,'
    # returns the coefficient of determination, R^2, for the model that produced 
    # predictions.
    # 
    # Numpy has a couple of functions -- np.mean() and np.sum() --
    # that you might find useful, but you don't have to use them.

    # YOUR CODE GOES HERE

    fi_square = np.square(data-predictions).sum()
    mean_y = data.mean()
    y_square = np.square(data-mean_y).sum()

    r_squared = 1- (fi_square/y_square)

    return r_squared