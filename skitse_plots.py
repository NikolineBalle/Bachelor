import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy import optimize
from scipy.special import erfinv
from scipy.interpolate import interp1d
from scipy import interpolate

def poly_interp(xi_multi,yi_multi):
    """
    General polynomial interpolation. 

    Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2,...,n-1
    where n = len(xi) = len(yi).

    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2 + ... + c[N-1]*x**(N-1).

    """
    # check inputs and print error message if not valid:
    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi_multi) is np.ndarray) and (type(yi_multi) is np.ndarray), error_message

    error_message = "xi and yi should have the same length "
    assert len(xi_multi)==len(yi_multi), error_message

    # The linear system to interpolate through data points:
    # Uses a list comprehension
    
    n = len(xi_multi)
    A = np.vstack([xi_multi**j for j in range(n)]).T
    c = solve(A,yi_multi)

    return c

def plot_poly(xi_multi, yi_multi, c):
    """
    Plot the resulting function along with the data points.
    """
    x = np.linspace(xi_multi.min() - 1,  xi_multi.max() + 1, 1000)

    # Using Horner's rule for defining interpolating polynomial:
    n = len(xi_multi)
    y = c[n-1]
    for j in range(n-1, 0, -1):
        y = y*x + c[j-1]
        
    ## Plotting
    plt.figure()
    plt.clf()
    plt.plot(xi_multi,yi_multi,'c.', markersize=15) 
    plt.plot(21, 1.3, '.', color='black', markersize=15)
    plt.plot(x,y,'b-')
    plt.ylim(yi_multi.min()-1, yi_multi.max()+1)
    plt.legend(['Datapoints', 'Next datapoint', 'Interpolating polynomial'], loc='best', numpoints=1, prop={'size': 16})

    plt.show()

def plot_splines(x,y, datapoint):
    """
    Input: x and y values that the splines should interpolate. 
    Next datapoint that splines doesn't interpolate with.

    Returns: Plot of (x,y) values, linear and cubic splines interpolating these and an extra datapoint. 
    """
    f1 = interp1d(x, y)
    f2 = interp1d(x, y, kind='cubic')
    xnew = np.linspace(0, 20, num=41, endpoint=True)

    plt.figure()
    plt.plot(x, y, '.', color = 'c', markersize=15)
    plt.plot(datapoint[0], datapoint[1], '.', color = 'black', markersize=15)
    plt.plot(xnew, f1(xnew), '-', xnew, f2(xnew), '--')
    plt.legend(['Datapoints', 'Next datapoint', 'Linear', 'Cubic'], loc='best', numpoints=1, prop={'size': 16})
    plt.xlim(0, 25)
    plt.ylim(-1.5, 1.5)
    plt.show()
    
def sum_of_functions(x,y, x1, y1, point1, point2):
    """
    Input: 
    (x,y) coordinates that the first polynomial should interpolate.
    (x,y) coordinates that the second polynomial should interpolate.
    Start and ending point for the two functions.
    
    Returns:
    Plot of the two functions from the start point to the end point, and the sum of the functions.
    """
    xnew = np.linspace(1,10,75)
    f2 = interp1d(x, y, kind = 'cubic')
    f3 = interp1d(x1, y1, kind = 'cubic')

    plt.figure()
    plt.plot(point1[0], point1[1], '.', color='black', markersize=18)
    plt.plot(point2[0], point2[1], 'c.', markersize=18)
    plt.plot(xnew, f2(xnew), 'c-')
    plt.plot(xnew, f3(xnew), 'b-', color='blue')
    plt.plot(xnew, f3(xnew)+f2(xnew), '-', color='black')
    plt.legend(['$(x_{a},A)$', '$(x_{b},B)$', '$g(x)$', '$y(x)$',  '$\widetilde{y}(x)$'], loc='best', numpoints=1, prop={'size': 16})
    plt.xlim(0, 11)
    plt.ylim(0, 10)
    plt.show()


# Plot of linear and cubic splines interpolating x and y coordinates and an extra datapoint
x = np.linspace(0, 20, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
datapoint = [21, 1.3]
plot_splines(x,y, datapoint)

# Interpolating polynomial through x and y coordinates
c = poly_interp(x, y)
plot_poly(x, y, c)

# Plot of two functions and the sum of them going from start to end point
x = [1, 3, 5, 8, 10]
y = [0, 2, 1, 4, 0]
x1 = [1, 3, 5, 8, 10]
y1 = [3, 3, 3, 5, 6]
point1 = [1, 3]
point2 = [10, 6]
sum_of_functions(x,y, x1, y1, point1, point2)







