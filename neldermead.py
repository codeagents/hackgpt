import numpy as np

def nelder_mead(func, x0, alpha=1, gamma=2, rho=0.5, sigma=0.5, max_iter=1000, tol=1e-6):
    n = len(x0)
    x0 = np.asarray(x0)
    
    # Initialize simplex vertices
    simplex = [x0]
    for i in range(n):
        x = x0.copy()
        x[i] += 0.1 * (x0[i] if x0[i] != 0 else 1)
        simplex.append(x)
    
    simplex_fvals = [func(x) for x in simplex]

    for iteration in range(max_iter):
        # Sort the simplex vertices and their function values
        order = np.argsort(simplex_fvals)
        simplex = [simplex[i] for i in order]
        simplex_fvals = [simplex_fvals[i] for i in order]

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

        if np.linalg.norm(simplex[0] - simplex[-1]) < tol:
            break

    return simplex[0], simplex_fvals[0]

# Example usage:
def black_box_function(x):
    return (x[0] - 3) ** 2 + (x[1] + 2) ** 2

x0 = [0, 0]
max_iter = 1000
tol = 1e-6

optimal_x, optimal_fval = nelder_mead(black_box_function, x0, max_iter=max_iter, tol=tol)
print(f"Optimal solution: {optimal_x}, function value: {optimal_fval}")
