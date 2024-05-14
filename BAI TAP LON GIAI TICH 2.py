
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

#Below is the 'parse_equation' function that helps converts standard mathematical functions to NumPy functions so that these functions can be used in operations with NumPy.
def parse_equation(equation): 
    equation = re.sub(r'\bsin\b', 'np.sin', equation)   #For example, this line replaces 'sin' with 'np.sin'.
    equation = re.sub(r'\bcos\b', 'np.cos', equation)
    equation = re.sub(r'\btan\b', 'np.tan', equation)
    equation = re.sub(r'\barcsin\b', 'np.arcsin', equation)
    equation = re.sub(r'\barccos\b', 'np.arccos', equation)
    equation = re.sub(r'\barctan\b', 'np.arctan', equation)
    equation = re.sub(r'\bsqrt\b', 'np.sqrt', equation)
    equation = re.sub(r'\bexp\b', 'np.exp', equation)
    equation = re.sub(r'\blog\b', 'np.log', equation)
    equation = re.sub(r'\babs\b', 'np.abs', equation)
    return equation

#Plot a 3D surface graph of a two-variable function and Mark the extremum points (both minima and maxima) on the surface.
def plot_surface(f, x_range, y_range):
    fig = plt.figure()                                  #Create a new figure object to hold the plot
    ax = fig.add_subplot(111, projection='3d')          #Add a subplot to the created figure object with a 3D configuration.
    X = np.arange(x_range[0], x_range[1], 0.1)          #Create two arrays of values X and Y from the x_range and y_range with a step size of 0.1. These are grid values on the x and y axes.
    Y = np.arange(y_range[0], y_range[1], 0.1)          
    X, Y = np.meshgrid(X, Y)                            #Create a 2D mesh grid from the X and Y arrays. Each point on this grid corresponds to a pair of (x, y) values.
    Z = f(X, Y)                                         #Calculate the values of Z for each pair of (X,Y) by applying the function f.
    ax.plot_surface(X, Y, Z, cmap='viridis')            #Plot a 3D surface on the grid (X, Y, Z) with colors specified by the Viridis color maps


#Call the extrema function to find the extrema (minima and maxima) of the function f within the range of x and y values.
    extrema_points = extrema(f, x_range, y_range)
    for point in extrema_points:                                #Iterate through the list of extrema points.
        x, y, type_extrema = point
        if type_extrema == "Minimum":
            ax.scatter([x], [y], [f(x, y)], color='g', s=50)    #If it's a minimum, draw a green (g) point at that location
        elif type_extrema == "Maximum":
            ax.scatter([x], [y], [f(x, y)], color='r', s=50)    #If it's a maximum, draw a red (r) point.

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

#The 'gradient' function approximates the gradient of a two-variable function f at a point (x, y).
#Use the method of approximating derivatives by computing partial derivatives with respect to x and y using a small interval h.
def gradient(f, x, y, h=0.01):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])


#The 'hessian'  function calculates the Hessian matrix of a two-variable function f at a point (x, y)  using second-order approximations of derivatives with a small interval h.
def hessian(f, x, y, h=0.01):
    df_dxdx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    df_dydy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    df_dxdy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return np.array([[df_dxdx, df_dxdy], [df_dxdy, df_dydy]])


#The 'extrema' function is written to find the extrema points of a two-variable function f within a specified range of x and y. The extrema points include minima, maxima, and saddle points.
def extrema(f, x_range, y_range, threshold=1e-5):
    extrema = []
    for x in np.arange(x_range[0], x_range[1], 0.1):
        for y in np.arange(y_range[0], y_range[1], 0.1):
            grad = gradient(f, x, y)                            #Calculate the gradient (grad) and the Hessian matrix (hess) of the function f at the point (x, y) by calling the 'gradient' and 'hessian' function.
            hess = hessian(f, x, y)
            eigvals, _ = np.linalg.eig(hess)                    #Calculate the eigenvalues (eigvals) of the Hessian matrix hess using 'np.linalg.eig'. These eigenvalues will be used to determine the type of extremum point (minimum, maximum, or saddle point).
            if all(eigval > 0 for eigval in eigvals):           #If all the eigenvalues of the Hessian matrix are greater than 0 (i.e., the matrix is positive definite), and the magnitude of the gradient is smaller than the threshold, then (x, y) is the minimum point.
                if np.linalg.norm(grad) < threshold:
                    extrema.append((x, y, "Minimum"))
            elif all(eigval < 0 for eigval in eigvals):         #If all the eigenvalues of the Hessian matrix are less than 0 (i.e., the matrix is negative definite), and the magnitude of the gradient is smaller than the threshold, then (x, y) is a maximum point.
                if np.linalg.norm(grad) < threshold:
                    extrema.append((x, y, "Maximum"))
            else:
                extrema.append((x, y, "Saddle"))                #If it is neither a minimum nor a maximum (i.e., the eigenvalues have different signs), then (x, y) is a saddle point and is added to the extrema list with the type "Saddle". 
    return extrema

if __name__ == "__main__":

    equation = input("Enter the function equation in terms of x and y: ")
    equation = parse_equation(equation)
    print(equation)

    x_range = (-5, 5)
    y_range = (-5, 5)

    f = lambda x, y: eval(equation)

    plot_surface(f, x_range, y_range)