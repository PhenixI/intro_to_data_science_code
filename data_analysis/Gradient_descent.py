import numpy
import pandas

def compute_cost(features, values, theta):
    """
    Compute the cost of a list of parameters, theta, given a list of features 
    (input data points) and values (output data points).
    """
    m = len(values)
    sum_of_square_errors = numpy.square(numpy.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)

    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    """

    # Write code here that performs num_iterations updates to the elements of theta.
    # times. Every time you compute the cost for a given list of thetas, append it 
    # to cost_history.
    # See the Instructor notes for hints. 
    
    m = len(values)
    cost_history = []

    pre_y = numpy.dot(features,theta)
    for i in range(num_iterations):
        theta = theta - (alpha/float(m))*numpy.dot(pre_y-values,features)
        cur_y = numpy.dot(features,theta)
        cost = numpy.square(cur_y - values).sum()/(2.0*m)
        cost_history.append(cost)
        pre_y = cur_y
    ###########################
    ### YOUR CODE GOES HERE ###
    ###########################

    return theta, pandas.Series(cost_history) # leave this line for the grader
