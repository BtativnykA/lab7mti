import random
import math

# 1. ГЕНЕРАЦІЯ ДАНИХ
def generate_data(n=100):
    """Генерує матрицю A з діагональним переважанням та вектор B."""
    A = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i != j:
                A[i][j] = random.uniform(-1.0, 1.0)
                row_sum += abs(A[i][j])
        # Забезпечуємо строге діагональне переважання
        A[i][i] = row_sum + random.uniform(10.0, 20.0)
        
    X_true = [2.5] * n
    
    # Обчислюємо B = A * X_true
    B = [sum(A[i][j] * X_true[j] for j in range(n)) for i in range(n)]
    
    # Записуємо у файли
    with open("matrix_A.txt", "w") as f:
        for row in A:
            f.write(" ".join(str(x) for x in row) + "\n")
            
    with open("vector_B.txt", "w") as f:
        for x in B:
            f.write(str(x) + "\n")
            
    print(f"Згенеровано матрицю A ({n}x{n}) з діагональним переважанням.")
    print("Точний розв'язок: всі x_i = 2.5\n")
    return A, B, X_true

# 2. ЗЧИТУВАННЯ ДАНИХ
def read_matrix(filename):
    A = []
    with open(filename, "r") as f:
        for line in f:
            A.append([float(x) for x in line.split()])
    return A

def read_vector(filename):
    B = []
    with open(filename, "r") as f:
        for line in f:
            B.append(float(line.strip()))
    return B

# 3. ДОПОМІЖНІ ФУНКЦІЇ
def mat_vec_mul(A, X):
    """Множення матриці на вектор."""
    n = len(A)
    return [sum(A[i][j] * X[j] for j in range(n)) for i in range(n)]

def vec_norm(X):
    """Максимальна норма вектора (max |x_i|)."""
    return max(abs(x) for x in X)

def vec_diff_norm(X1, X2):
    """Норма різниці двох векторів (для умови збіжності)."""
    return max(abs(X1[i] - X2[i]) for i in range(len(X1)))

# 4. МЕТОДИ РОЗВ'ЯЗКУ
def simple_iteration(A, B, X0, eps, tau):
    """Метод простої ітерації."""
    n = len(B)
    X = X0[:]
    iterations = 0
    
    while True:
        AX = mat_vec_mul(A, X)
        X_new = [X[i] - tau * (AX[i] - B[i]) for i in range(n)]
        
        diff = vec_diff_norm(X_new, X)
        X = X_new
        iterations += 1
        
        if diff < eps:
            return X, iterations

def jacobi(A, B, X0, eps):
    """Метод Якобі."""
    n = len(B)
    X = X0[:]
    iterations = 0
    
    while True:
        X_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * X[j] for j in range(n) if j != i)
            X_new[i] = (B[i] - s) / A[i][i]
            
        diff = vec_diff_norm(X_new, X)
        X = X_new
        iterations += 1
        
        if diff < eps:
            return X, iterations

def seidel(A, B, X0, eps):
    """Метод Зейделя (Гауса-Зейделя)."""
    n = len(B)
    X = X0[:]
    iterations = 0
    
    while True:
        X_old = X[:]
        for i in range(n):
            s1 = sum(A[i][j] * X[j] for j in range(i))
            s2 = sum(A[i][j] * X_old[j] for j in range(i + 1, n))
            X[i] = (B[i] - s1 - s2) / A[i][i]
            
        diff = vec_diff_norm(X, X_old)
        iterations += 1
        
        if diff < eps:
            return X, iterations

def main():
    n = 100
    eps = 1e-14
    
    print("=" * 60)
    print("ЛАБОРАТОРНА РОБОТА №7")
    print("Ітераційні методи розв'язку СЛАР")
    print("=" * 60)
    
    print("\nПУНКТ 1: Генерація даних")
    A, B, X_true = generate_data(n)
    
    print("ПУНКТ 2-3: Зчитування даних та початкове наближення")
    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")
    X0 = [1.0] * n
    
    # Вибір tau для простої ітерації. 
    # Оскільки матриця з діагональним переважанням, оберемо безпечне tau = 1 / max(a_ii)
    max_diag = max(A[i][i] for i in range(n))
    tau = 1.0 / max_diag
    print(f"Параметр tau для простої ітерації: {tau:.6e}\n")
    
    print("ПУНКТ 4: Розв'язок методами")
    
    # 1. Метод простої ітерації
    X_si, it_si = simple_iteration(A, B, X0, eps, tau)
    err_si = max(abs(X_si[i] - 2.5) for i in range(n))
    
    # 2. Метод Якобі
    X_jac, it_jac = jacobi(A, B, X0, eps)
    err_jac = max(abs(X_jac[i] - 2.5) for i in range(n))
    
    # 3. Метод Зейделя
    X_sei, it_sei = seidel(A, B, X0, eps)
    err_sei = max(abs(X_sei[i] - 2.5) for i in range(n))
    
    print("\nРезультати обчислень:")
    print("-" * 75)
    print(f"{'Метод':<25} | {'Кількість ітерацій':>20} | {'Похибка max|x_i - 2.5|':>25}")
    print("-" * 75)
    print(f"{'Проста ітерація':<25} | {it_si:>20} | {err_si:>25.6e}")
    print(f"{'Якобі':<25} | {it_jac:>20} | {err_jac:>25.6e}")
    print(f"{'Зейделя':<25} | {it_sei:>20} | {err_sei:>25.6e}")
    print("-" * 75)
    
    print("\n" + "=" * 60)
    print("РОБОТА ВИКОНАНА")
    print("=" * 60)

if __name__ == "__main__":
    main()