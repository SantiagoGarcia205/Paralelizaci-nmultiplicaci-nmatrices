import random 
import time 

def sequential_matrix_multiplication(A, B): 
    rows_A = len(A) 
    cols_A = len(A[0]) 
    rows_B = len(B) 
    cols_B = len(B[0]) 

    if cols_A != rows_B: 
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación.") 
    # Inicializar la matriz resultado con ceros 
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)] 
    # Proceso de multiplicación "a pedal" 
    # Itera sobre las filas de A 
    for i in range(rows_A): 
        # Itera sobre las columnas de B 
        for j in range(cols_B): 
            # Itera sobre las filas de B (o columnas de A) para el producto punto 
            for k in range(cols_A): 
                C[i][j] += A[i][k] * B[k][j] 
    return C 

def generate_random_matrix(rows, cols): 
    matrix = [[random.random() for _ in range(cols)] for _ in range(rows)] 
    return matrix

if __name__ == "__main__": 
    # Pueden ajustar este valor si su máquina tiene más o menos recursos.
    # ¡Cuidado con valores muy grandes que puedan colgar su sistema!
    MATRIX_SIZE = 1500

    print(f"Generando matrices aleatorias de {MATRIX_SIZE}x{MATRIX_SIZE}...") 
    # Generar las dos matrices a multiplicar 
    matrix_A = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE) 
    matrix_B = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE) 

    print("Matrices generadas. Iniciando multiplicación secuencial...") 
    # Medir el tiempo de ejecución 
    start_time = time.time() 

    result_matrix = sequential_matrix_multiplication(matrix_A, matrix_B) 
    end_time = time.time() 

    elapsed_time = end_time - start_time 
    print("La multiplicación secuencial ha finalizado.") 
    print(f"Tiempo total de ejecución: {elapsed_time:.4f} segundos.")
