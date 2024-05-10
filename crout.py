import numpy as np
import scipy.linalg as linalg

def solusi_persamaan_linear_crout(matriks_A, vektor_b):

  try:
    P, L, U = linalg.lu(matriks_A, permute_l=False)
    y = linalg.solve_triangular(L, vektor_b, lower=True)
    x = linalg.solve_triangular(U, y)
    return x
  except ValueError as e:
    print(f"Matriks A singular. Tidak bisa melakukan dekomposisi LU: {e}")
    return None

matriks_A = np.array([[3, 1], [2, -1]])
vektor_b = np.array([1, 3])

answer = solusi_persamaan_linear_crout(matriks_A, vektor_b)

if answer is not None:
  print(f"Solusi Persamaan Linier dengan Metode Crout:\n{answer}")
else:
  print("Gagal menyelesaikan persamaan linier. Matriks A singular.")
