import numpy as np


def simplex_update(func, simplex, alpha=1, gamma=2, rho=0.5, sigma=0.5):
    """
    Perform a single iteration of the Nelder-Mead simplex update.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.
    simplex : ndarray
        The simplex with shape (n+1, n), where n is the number of dimensions.
    alpha : float, optional
        Reflection coefficient (default 1).
    gamma : float, optional
        Expansion coefficient (default 2).
    rho : float, optional
        Contraction coefficient (default 0.5).
    sigma : float, optional
        Shrink coefficient (default 0.5).

    Returns
    -------
    new_simplex : ndarray
        The updated simplex.
    """

    # Sort the simplex vertices by their function values
    sorted_indices = np.argsort([func(x) for x in simplex])
    simplex = simplex[sorted_indices]

    # Calculate the centroid of the n best vertices (excluding the worst vertex)
    centroid = np.mean(simplex[:-1], axis=0)

    # Perform reflection
    reflected = centroid + alpha * (centroid - simplex[-1])
    f_reflected = func(reflected)

    if func(simplex[0]) <= f_reflected < func(simplex[-2]):
        # Reflection is successful; replace the worst vertex with the reflected point
        simplex[-1] = reflected
    elif f_reflected < func(simplex[0]):
        # Perform expansion
        expanded = centroid + gamma * (reflected - centroid)
        f_expanded = func(expanded)

        if f_expanded < func(simplex[0]):
            # Expansion is successful; replace the worst vertex with the expanded point
            simplex[-1] = expanded
        else:
            # Expansion failed; use reflected point instead
            simplex[-1] = reflected
    else:
        # Perform contraction
        contracted = centroid + rho * (simplex[-1] - centroid)
        f_contracted = func(contracted)

        if f_contracted < func(simplex[-1]):
            # Contraction is successful; replace the worst vertex with the contracted point
            simplex[-1] = contracted
        else:
            # Perform shrink
            for i in range(1, len(simplex)):
                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])

    return simplex


def nelder_mead(func, x0, alpha=1, gamma=2, rho=0.5, sigma=0.5, tol=1e-6, max_iter=1000):
    """
    Minimize a scalar function using the Nelder-Mead algorithm.
    
    Parameters
    ----------
    func : callable
        The objective function to be minimized.
    x0 : array_like
        The initial guess.
    alpha : float, optional
        Reflection coefficient (default 1).
    gamma : float, optional
        Expansion coefficient (default 2).
    rho : float, optional
        Contraction coefficient (default 0.5).
    sigma : float, optional
        Shrink coefficient (default 0.5).
    tol : float, optional
        The convergence tolerance for function values (default 1e-6).
    max_iter : int, optional
        The maximum number of iterations to perform (default 1000).
    
    Returns
    -------
    x : ndarray
        The solution array.
    """
    # Initialize the simplex
    x0 = np.asarray(x0)
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        x = x0.copy()
        x[i] += 0.05 * x0[i] if x0[i] != 0 else 0.05
        simplex[i + 1] = x

    # Iterate until the maximum number of iterations or convergence
    for _ in range(max_iter):
        # Update the simplex using the simplex update function
        simplex = simplex_update(func, simplex, alpha, gamma, rho, sigma)

        # Check for convergence
        if np.abs(func(simplex[0]) - func(simplex[-1])) < tol:
            break

    return simplex[0]


# Test function
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


# Test the Nelder-Mead algorithm
x0 = [0.8, 1.2]
optimal_x = nelder_mead(rosenbrock, x0)
print(f"Optimal solution: {optimal_x}")
