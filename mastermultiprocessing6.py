import random
import time
import multiprocessing 

def worker(cols_B,cols_A,A,B,i_start, i_end,out_q):
        bloque=[]
        # Proceso de multiplicación "a pedal" 
        # Itera sobre las filas de A 
        for i in range(i_start, i_end): 
            # Itera sobre las columnas de B
            fila_res= [0.0] * cols_B #->para que comience la fila en 0 sobre todas las columnas de b para que sea el resultado.
            for j in range(cols_B): 
                s=0.0
                # Itera sobre las filas de B (o columnas de A) para el producto punto 
                for k in range(cols_A): 
                    s+= A[i][k] * B[k][j]
                fila_res[j]=s #de esta manera s el número que va en la columna j de la fila i de C.
            bloque.append((i,fila_res)) # donde va la fila, columna y el resultado corresponidente
        out_q.put(bloque) # entrega todos los bloques


def sequential_matrix_multiplication(A, B): 
    rows_A = len(A) 
    cols_A = len(A[0]) 
    rows_B = len(B) 
    cols_B = len(B[0]) 

    if cols_A != rows_B: 
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación.") 

    # Inicializar la matriz resultado con ceros 
    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)] 

            
    # División simple en dos procesos por filas
    mid = rows_A // 6
    q = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=worker, args=(cols_B,cols_A,A,B,0,mid,q))
    p2 = multiprocessing.Process(target=worker, args=(cols_B,cols_A,A,B,mid,mid*2,q))
    p3 = multiprocessing.Process(target=worker, args=(cols_B,cols_A,A,B,mid*2,mid*3,q))
    p4 = multiprocessing.Process(target=worker, args=(cols_B,cols_A,A,B,mid*3,mid*4,q))
    p5 =multiprocessing.Process(target=worker, args=(cols_B,cols_A,A,B,mid*4,mid*5,q))
    p6 =multiprocessing.Process(target=worker, args=(cols_B,cols_A,A,B,mid*5,rows_A,q))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()

    bloque1 = q.get()
    bloque2 = q.get()
    bloque3 = q.get()
    bloque4= q.get()
    bloque5 = q.get()
    bloque6= q.get()


    for i, fila in bloque1:
        C[i] = fila
    for i, fila in bloque2:
        C[i] = fila
    for i, fila in bloque3:
        C[i] = fila
    for i, fila in bloque4:
        C[i] = fila
    for i, fila in bloque5:
        C[i] = fila
    for i, fila in bloque6:
        C[i] = fila

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    return C

def generate_random_matrix(rows, cols): 
    return [[random.random() for _ in range(cols)] for _ in range(rows)]

if __name__ == "__main__": 
    MATRIX_SIZE = 1500

    print(f"Generando matrices aleatorias de {MATRIX_SIZE}x{MATRIX_SIZE}...") 
    matrix_A = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE) 
    matrix_B = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE) 

    print("Matrices generadas. Iniciando multiplicación por multiprocessing...") 
    start_time = time.time() 

    result_matrix = sequential_matrix_multiplication(matrix_A, matrix_B) 

    end_time = time.time() 
    elapsed_time = end_time - start_time 
    print("La multiplicación con multiprocessing ha finalizado.") 
    print(f"Tiempo total de ejecución: {elapsed_time:.4f} segundos.")
