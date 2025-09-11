import sys
print(">>> Python:", sys.executable)

from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

def generate_random_matrix(rows, cols, seed=None):
    if seed is not None:
        random.seed(seed)
    return [[random.random() for _ in range(cols)] for _ in range(rows)]

def mult_listas(A_part, B):

    if not A_part:  # por si a algún rank le tocan 0 filas
        return []
    rows_A = len(A_part)
    cols_A = len(A_part[0])
    rows_B = len(B)
    cols_B = len(B[0])
    if cols_A != rows_B:
        raise ValueError(f"Dimensiones incompatibles: A_part=({rows_A},{cols_A}), B=({rows_B},{cols_B})")

    C_part = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            s = 0.0
            for k in range(cols_A):
                s += A_part[i][k] * B[k][j]
            C_part[i][j] = s
    return C_part

def mpi_matrix_multiplication(A, B, comm=MPI.COMM_WORLD, root=0):
    """Multiplica A x B con MPI repartiendo filas de A. 
       A y B se pasan en root; en otros ranks deben ser None.
       Devuelve C en root; en otros ranks devuelve None.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        N = len(A)
        K = len(A[0])
        Kb = len(B)        
        M = len(B[0])
        if K != Kb:
            raise ValueError(f"Dimensiones incompatibles: A es ({N},{K}) y B es ({Kb},{M})")
        base = N // size
        resto = N % size
        partes = []
        start = 0
        for r in range(size):
            extra = 1 if r < resto else 0
            end = start + base + extra
            partes.append(A[start:end])  
            start = end
    else:
        N = K = M = None
        partes = None

    B = comm.bcast(B, root=root)

    A_local = comm.scatter(partes, root=root)

    C_local = mult_listas(A_local, B)

    recolectadas = comm.gather(C_local, root=root)

    if rank == root:
        C = []
        for bloque in recolectadas:
            C.extend(bloque)
        return C
    else:
        return None

if __name__ == "__main__":
    MATRIX_SIZE = 1500

    if rank == root:
        print(f"[root] Generando matrices aleatorias de {MATRIX_SIZE}x{MATRIX_SIZE}...")
        A = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE)  # A: N x K
        B = generate_random_matrix(MATRIX_SIZE, MATRIX_SIZE)  # B: K x M
        print("[root] Matrices generadas. Iniciando multiplicación con MPI...")
    else:
        A = None
        B = None

    comm.Barrier()
    t0 = MPI.Wtime()

    C = mpi_matrix_multiplication(A, B, comm=comm, root=root)

    comm.Barrier()
    t1 = MPI.Wtime()

    if rank == root:
        n_filas = len(C)
        n_cols = len(C[0]) if C else 0
        print("[root] La multiplicación con MPI ha finalizado.")
        print(f"[root] C tiene forma ({n_filas},{n_cols})")
        print(f"[root] Tiempo total (incl. comunicación): {t1 - t0:.4f} s")
