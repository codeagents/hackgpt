"""
{
  "suggestions": [
    {
      "code": "def initialize_simplex(x0, scale=0.1):",
      "comment": "Add a scale parameter to the initialize_simplex function to allow for more flexibility in the initial simplex size."
    },
    {
      "code": "x[i] += scale * (x0[i] if x0[i] != 0 else 1)",
      "comment": "Use the scale parameter in the loop to adjust the size of the initial simplex."
    },
    {
      "code": "def nelder_mead(func, x0, alpha=1, gamma=2, rho=0.5, sigma=0.5, max_iter=1000, tol=1e-6, scale=0.1):",
      "comment": "Add a scale parameter to the nelder_mead function to pass it to the initialize_simplex function."
    },
    {
      "code": "simplex = initialize_simplex(x0, scale=scale)",
      "comment": "Pass the scale parameter to the initialize_simplex function when calling it in the nelder_mead function."
    },
    {
      "code": "if np.linalg.norm(simplex[0] - simplex[-1]) < tol:",
      "comment": "Consider using np.allclose with the atol parameter instead of np.linalg.norm for checking convergence."
    },
    {
      "code": "if np.allclose(simplex[0], simplex[-1], atol=tol):",
      "comment": "Replace the np.linalg.norm line with np.allclose for checking convergence."
    },
    {
      "code": "def nelder_mead(func, x0, alpha=1, gamma=2, rho=0.5, sigma=0.5, max_iter=1000, tol=1e-6, scale=0.1, callback=None):",
      "comment": "Add a callback parameter to the nelder_mead function to allow for custom actions during the optimization process."
    },
    {
      "code": "if callback is not None: callback(simplex, simplex_fvals, iteration)",
      "comment": "Call the callback function with the current simplex, function values, and iteration number if it is provided."
    }
  ]
}
"""
import numpy as np

# Added scale parameter for more flexibility in initial simplex size
def initialize_simplex(x0, scale=0.1):
    n = len(x0)
    x0 = np.asarray(x0)

    simplex = [x0]
    for i in range(n):
        x = x0.copy()
        # Used scale parameter to adjust the size of the initial simplex
        x[i] += scale * (x0[i] if x0[i] != 0 else 1)
        simplex.append(x)

    return simplex


# Example usage:

x0 = [0, 0]
simplex = initialize_simplex(x0)
print(f'Simplex vertices: {simplex}')


def sort_simplex(simplex, func):
    simplex_fvals = [func(x) for x in simplex]
    order = np.argsort(simplex_fvals)
    sorted_simplex = [simplex[i] for i in order]
    sorted_simplex_fvals = [simplex_fvals[i] for i in order]
    return sorted_simplex, sorted_simplex_fvals

def update_simplex(simplex, simplex_fvals, func, alpha=1, gamma=2, rho=0.5, sigma=0.5):
    n = len(simplex) - 1
    centroid = np.mean(simplex[:-1], axis=0)
    xr = centroid + alpha * (centroid - simplex[-1])
    fxr = func(xr)

    if fxr < simplex_fvals[0]:
        xe = centroid + gamma * (xr - centroid)
        fxe = func(xe)
        if fxe < simplex_fvals[0]:
            simplex[-1] = xe
            simplex_fvals[-1] = fxe
        else:
            simplex[-1] = xr
            simplex_fvals[-1] = fxr
    elif fxr < simplex_fvals[-2]:
        simplex[-1] = xr
        simplex_fvals[-1] = fxr
    else:
        if fxr < simplex_fvals[-1]:
            simplex[-1] = xr
            simplex_fvals[-1] = fxr

        xc = centroid + rho * (simplex[-1] - centroid)
        fxc = func(xc)

        if fxc < simplex_fvals[-1]:
            simplex[-1] = xc
            simplex_fvals[-1] = fxc
        else:
            for i in range(1, n + 1):
                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                simplex_fvals[i] = func(simplex[i])

    return simplex, simplex_fvals

# Added scale and callback parameters to nelder_mead function
def nelder_mead(func, x0, alpha=1, gamma=2, rho=0.5, sigma=0.5, max_iter=1000, tol=1e-6, scale=0.1, callback=None):
    # Passed scale parameter to initialize_simplex function
    simplex = initialize_simplex(x0, scale=scale)
    simplex_fvals = [func(x) for x in simplex]

    for iteration in range(max_iter):
        simplex, simplex_fvals = sort_simplex(simplex, func)
        simplex, simplex_fvals = update_simplex(simplex, simplex_fvals, func, alpha, gamma, rho, sigma)

        # Replaced np.linalg.norm with np.allclose for checking convergence
        if np.allclose(simplex[0], simplex[-1], atol=tol):
            break

        # Call the callback function with the current simplex, function values, and iteration number if provided
        if callback is not None:
            callback(simplex, simplex_fvals, iteration)

    return simplex[0], simplex_fvals[0]


# Example usage:
def black_box_function(x):
    return (x[0] - 3) ** 2 + (x[1] + 2) ** 2

x0 = [0, 0]
max_iter = 1000
tol = 1e-6

# Added a simple callback function to print the current iteration number
def print_iteration(simplex, simplex_fvals, iteration):
    print(f"Iteration {iteration}")

optimal_x, optimal_fval = nelder_mead(black_box_function, x0, max_iter=max_iter, tol=tol, callback=print_iteration)
print(f"Optimal solution: {optimal_x}, function value: {optimal_fval}")
