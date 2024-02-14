import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as la

# Define parameters
lambd = 0.1
mu = 1
rho = 1

# omega_list = [4, 8.25, 16.5, 33, 66.5]
# omega_list = [25, 50, 100, 200]
omega = 4

kappa_p = rho * omega**2 / (lambd + 2 * mu)
kappa_s = rho * omega**2 / mu


# Section 2.2

# Define I, K, Kinv
Id = np.identity(6)

K = np.diag(np.array([kappa_p, kappa_s, kappa_s]))
Kinv = np.diag(np.array([1 / kappa_p, 1 / kappa_s, 1 / kappa_s]))

# Introduce A
A1 = np.block([[np.zeros((3, 3)), 1 / 1j * Kinv], [1j * K, np.zeros((3, 3))]])

# There holds that A^2 = I
print("A^2 = Id: ", np.allclose(np.linalg.norm(np.dot(A1, A1) - Id), 0))

# Introduce C such as in (10)
C1 = 1 / 2 * (Id + A1)

# Introduce C with the coefficients such as in (9)
C = (
    1
    / 2
    * np.array(
        [
            [1, 0, 0, 1 / 1j / kappa_p, 0, 0],
            [0, 1, 0, 0, 1 / 1j / kappa_s, 0],
            [0, 0, 1, 0, 0, 1 / 1j / kappa_s],
            [1j * kappa_p, 0, 0, 1, 0, 0],
            [0, 1j * kappa_s, 0, 0, 1, 0],
            [0, 0, 1j * kappa_s, 0, 0, 1],
        ]
    )
)


print("C1 = C", np.allclose(np.linalg.norm(C1 - C), 0))

# Check that C is a projector
print("C^2 = C", np.allclose(np.linalg.norm(np.dot(C, C) - C), 0))

# Section 2.3 (Two subdomains)

sigma_1 = 1
sigma_2 = 1

X = np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), -np.eye(3)]])


# MTF operator

MTF = np.block(
    [[(1 + sigma_1) * Id - C, -sigma_1 * X], [-sigma_2 * X, (1 + sigma_2) * Id - C]]
)


# Block Jacobi iteration matrix

D = np.block([[(1 + sigma_1) * Id - C, 0 * X], [0 * X, (1 + sigma_2) * Id - C]])


D_inv = np.linalg.inv(D)

T = np.block([[0 * Id, sigma_1 * X], [sigma_2 * X, 0 * Id]])


J = np.dot(D_inv, T)

MTF2 = np.dot(MTF, MTF)

# Eigenvalues distribution

e_MTF, vect = la.eig(MTF)
e_MTF2, vect = la.eig(MTF2)
e_J, vect = la.eig(J)

a, b = np.sqrt(sigma_1 / (sigma_1 + 1)), np.sqrt(sigma_2 / (sigma_2 + 1))

sigma_J = np.array([-a, -b, a, b])

fig, ax = plt.subplots()

ax.scatter(
    e_MTF.real, 0 * e_MTF.imag, s=50, facecolors="none", edgecolor="r", label="MTF"
)
ax.scatter(
    e_MTF2.real, 1 * np.ones(12), s=50, facecolors="none", edgecolor="b", label=r"MTF^2"
)
ax.scatter(
    e_J.real, 2 * np.ones(12), s=50, facecolors="none", edgecolor="orange", label="J"
)
ax.scatter(sigma_J, 2 * np.ones(4), s=50, color="k", marker="x")
ax.legend()
plt.savefig("eigval.pdf")

sigma = np.arange(101) / 100 
a = np.sqrt(sigma / (sigma + 1))
fig = plt.figure()
plt.plot(sigma, a)

plt.plot(sigma, a)
plt.grid()
plt.show(block=False)
plt.savefig("sigma.pdf")
