import numpy as np
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))
print("Enter elements of Matrix A:")
A = np.array([[int(input()) for j in range(cols)] for i in range(rows)])
print("Enter elements of Matrix B:")
B = np.array([[int(input()) for j in range(cols)] for i in range(rows)])
print("\nMatrix Addition (A + B):")
print(A + B)
print("\nMatrix Subtraction (A - B):")
print(A - B)
if cols == rows:
    print("\nMatrix Multiplication (A × B):")
    print(np.matmul(A, B))
else:
    print("\nMatrix Multiplication is not possible.")
print("\nTranspose of Matrix A:")
print(A.T)
if rows == cols:
    if np.linalg.det(A) != 0:
        print("\nInverse of Matrix A:")
        print(np.linalg.inv(A))
    else:
        print("\nInverse does not exist (Determinant is 0).")
else:
    print("\nInverse is not possible (Matrix is not square).")
