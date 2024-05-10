import numpy as np
import scipy.linalg as linalg

def solusi_persamaan_lu_gauss(matriks_A, vektor_b):

  try:
    P, L, U = linalg.lu(matriks_A)
    y = linalg.solve_triangular(L, vektor_b, lower=True)
    x = linalg.solve_triangular(U, y)
    return x
  except ValueError as e:
    print(f"Matriks A singular, dekomposisi LU gagal: {e}")
    return None

matriks_A = np.array([[1, 0, 1], [2, 3, 5], [2, 2, -3]])
vektor_b = np.array([-4, 2, 31])

answer = solusi_persamaan_lu_gauss(matriks_A, vektor_b)

if answer is not None:
  print(f"Solusi persamaan linier dengan dekomposisi LU Gauss:\n{answer}")
else:
  print("Gagal menyelesaikan persamaan linier; Matriks A singular.")