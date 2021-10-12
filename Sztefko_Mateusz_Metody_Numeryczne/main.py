import math
import numpy as np



def cylinder_area(r:float,h:float):
    if (r < 0) or (h < 0):
        return np.NaN
    
    else:
        result = ((2*math.pi*r**2) + (2*math.pi*r*h))
        
        return result

V_1 = np.arange(2,5,0.2)

V_2 = np.linspace(2,5,16)


def fib(n):
    
    list_of_Fib = []
    if not isinstance(n,int):
        return None
    elif n <= 0:
        return None
    elif n == 1:
        return [1]
    else:

        list_of_Fib.append(1)
        last, value = 0, 1
    
        for i in range(0, n-1):
            last, value = value, last + value
            list_of_Fib.append(float(value))

    array_of_Fib = np.array([list_of_Fib])
    return array_of_Fib


# print(fib(15))
    # """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego. 
    # Szczegółowy opis w zadaniu 3.
    
    # Parameters:
    # n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    # Returns:
    # np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    # """
    

def matrix_calculations(a:float) -> tuple[np.array, np.array, float]:
    M:np.array = np.array([[a,1,-a], [0,1,1], [-a,a,1]])  

    Mdet:float = np.linalg.det(M)

    Minv:np.array = np.zeros(3) 
    if Mdet != 0:
        Minv:np.array = np.linalg.inv(M)
    else:
        Minv = np.NaN
    
    Mt:np.array = np.transpose(M)
    
    return Minv, Mt, Mdet
    
    
    # """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    # na podstawie parametru a.  
    # Szczegółowy opis w zadaniu 4.
    
    # Parameters:
    # a (float): wartość liczbowa 
    
    # Returns:
    # touple: krotka zawierająca wyniki obliczeń 
    # (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    # """
    
    



def From_matrix():
    M = np.array([[3, 1, -2, 4],[0, 1, 1, 5],[-2, 1, 1, 6],[4, 3, 0, 1]])
    w1 = M[:,2]
    w2 = M[1,:]
    
    print(M[0,0])
    print(M[2,2])
    print(M[2,1])
    print(w1)
    print(w2)
    
#print(From_matrix())    




def custom_matrix(m:int, n:int):

    if (m > 0) and isinstance(m, int) and (n > 0) and isinstance(n, int):
        
        Matrix = np.zeros((m, n))
        r_index:int = 0
        c_index:int = 0     
        
        for i in range(n):
            for j in range(m):

                if r_index > c_index:
                    Matrix[r_index, c_index]= r_index

                else:
                    Matrix[r_index, c_index]= c_index

                r_index += 1
            r_index = 0
            c_index += 1       
        return Matrix

    else:
        return None
m = np.random.randint(3, 7)
n = np.random.randint(3, 7)
print(custom_matrix(m, n))



# def custom_matrix_numpy(m:int, n:int):







    # """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    # z opisem zadania 7.  
    
    # Parameters:
    # m (int): ilość wierszy macierzy
    # n (int): ilość kolumn macierzy  
    
    # Returns:
    # np.ndarray: macierz zgodna z opisem z zadania 7.
    # """
def Task7_8():

    v1:np.array = np.array([[1],[3],[13]])
    v2:np.array = np.array([[8],[5],[-2]])

    v_4v1 = np.dot(4, v1)
    v_v2_add_vector = np.add(-v2, [[2],[2],[2]])
    v_v1_cauchy_v2:np.array = np.dot(v1, np.transpose(v2))
    v_v1_hadamard_v2:np = np.multiply(v1, v2)



    print(v_4v1)
    print(v_v2_add_vector)
    print(v_v1_cauchy_v2)
    print(v_v1_hadamard_v2)


    M1 = np.array([[1,-7,3],[-12,3,4],[5,13,-3]])

    Matrix_3M1 = np.dot(3, M1)
    Matrix_3M1_add_onematrix = np.add(np.dot(3, M1), np.ones(3))
    Matrix_M1transpose = np.transpose(M1)
    Matrix_M1v1 = np.dot(M1, v1)
    Matrix_v1transposeM1 = np.dot(np.transpose(v2), M1)

    print(Matrix_3M1)
    print(Matrix_3M1_add_onematrix)
    print(Matrix_M1transpose)
    print(Matrix_M1v1)
    print(Matrix_v1transposeM1)


print(Task7_8())

