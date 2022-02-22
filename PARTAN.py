import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def read_mat(path):
  x = loadmat(path)
  return x['Q'], x['b'], x['c']

def f(x, Q, b, c, mu):
    out = 0.5 * np.dot(x.T, np.dot(Q, x)) - np.dot(b.T,x) + 0.5 * mu *(np.dot(c.T, x))**2
    return out[0][0]

def df(x, Q, b, c, mu):
    out = np.dot(Q, x) - b + mu *np.dot(c, np.dot(c.T, x))
    return out

def df2(x, y):
    return np.array([[2, -5], [-5, 12*y*y]])

def get_solution(Q, b, c, mu):
    out = np.dot(np.linalg.inv(Q + mu * np.dot(c, c.T)), b)
    return out
                                 
def get_eigenvalues(Q):
    l = np.linalg.eigvals(Q)
    return l
    
def norm(x):
    return np.sqrt((x**2).sum())

def plot_error_partan(criterion, fx, d, mu): 
    plt.figure()

    plt.plot(range(len(criterion)), criterion)
    plt.yscale('log')
    plt.ylabel('Criterion ||df|| in log scale')
    plt.xlabel('k')
    plt.title('PARTAN algorithm criterion with mu={}'.format(mu))   
    plt.savefig("partan_error_hw6_mu={}.png".format(mu))   
    
    plt.figure()
    plt.plot(range(len(fx)), fx)
    plt.ylabel('f(x)')
    plt.xlabel('k')
    plt.title('PARTAN algorithm f(x) versus k with mu={}'.format(mu))   
    plt.savefig("partan_fx_hw6_mu={}.png".format(mu)) 

    plt.figure()

    plt.plot(range(len(d)), d)
    plt.yscale('log')
    plt.ylabel('|| xk - x*|| in log scale')
    plt.xlabel('k')
    plt.title('PARTAN algorithm || xk - x*|| versus k with mu={}'.format(mu))  
    plt.savefig("partan_distance_hw6_mu={}.png".format(mu)) 
    
def plot_error(criterion, fx, d, mu): 
    plt.figure()

    plt.plot(range(len(criterion)), criterion)
    plt.yscale('log')
    plt.ylabel('Criterion ||df|| in log scale')
    plt.xlabel('k')
    plt.title('Gradient descent algorithm criterion with mu={}'.format(mu))   
    plt.savefig("error_hw6_mu={}.png".format(mu))   
    
    plt.figure()
    plt.plot(range(len(fx)), fx)
    plt.ylabel('f(x)')
    plt.xlabel('k')
    plt.title('f(x) versus k with mu={}'.format(mu))   
    plt.savefig("fx_hw6_mu={}.png".format(mu)) 

    plt.figure()
    plt.plot(range(len(d)), d)
    plt.yscale('log')
    plt.ylabel('|| xk - x*|| in log scale')
    plt.xlabel('k')
    plt.title('|| xk - x*|| versus k with mu={}'.format(mu))   
    plt.savefig("distance_hw6_mu={}.png".format(mu)) 
    
def backtracking(xk, Q, b, c, mu, eta = 1.6, epsilon = 0.4):
    assert eta > 1
    assert 0 < epsilon < 0.5
    
    alpha = 1.0
    fx = f(xk, Q, b, c, mu)
    dfx = df(xk, Q, b, c, mu)
    dk = - dfx
    step = np.dot(dfx.T, dk)
    criterion = norm(dfx)

    while True:
        x_k1 = xk + alpha * dk
        fx_k1 = f(x_k1, Q, b, c, mu)
        if fx_k1 <= fx + epsilon * alpha * step:
            break
        else:
            alpha = alpha / eta
    #print(alpha)     
    return criterion, alpha, dk, fx    
    
def Gradient_Descent_Method(L, mu = 1000):
    Q, b, c = read_mat('HW6_data.mat')

    d = len(b)
    # Randomly intialize x_0
    k = 0 
    #xk = np.expand_dims(np.random.rand(d),-1)
    xk = np.expand_dims(np.zeros(d),-1)
    criterion = 10
       
    ## Determine the solution x_star
    x_star = get_solution(Q, b, c, mu)
  
    # Iterative process
    criterion_list = []
    fx_list = []
    d_list = []
    while criterion > L:
        criterion, alpha, dk, fx = backtracking(xk, Q, b, c, mu)
        xk = xk + alpha * dk
        #exit()
        print(criterion, alpha)
        criterion_list.append(criterion)
        fx_list.append(fx)
        d_list.append(norm(xk - x_star))
        k += 1
     
    plot_error(criterion_list, fx_list, d_list, mu)   

    print(len(criterion_list))
    
def Partan_Method(L, mu = 1000):
    Q, b, c = read_mat('HW6_data.mat')

    d = len(b)
    # Randomly intialize x_0
    k = 1
    #xk = np.expand_dims(np.random.rand(d),-1)
    xk = np.expand_dims(np.zeros(d),-1)
    criterion = 10
    
    ## Determine the solution x_star
    x_star = get_solution(Q, b, c, mu)
  
    # Iterative process
    # k = 1
    criterion, alpha, dk, fx = backtracking(xk, Q, b, c, mu)
    xk = xk + alpha * dk
    criterion_list = [criterion]
    fx_list = [fx]
    d_list = [norm(xk - x_star)]
        
    while criterion > L:
        _, alpha, dk, _ = backtracking(xk, Q, b, c, mu)
        yk = xk + alpha * dk
        _, beta, _, _ = backtracking(yk, Q, b, c, mu)
        xk = yk + beta * (yk - xk)
        ##
        fx = f(xk, Q, b, c, mu)
        dfx = df(xk, Q, b, c, mu)
        criterion = norm(dfx)
        #exit()
        print(criterion, alpha, beta)
        fx_list.append(fx)
        criterion_list.append(criterion)
        d_list.append(norm(xk - x_star))
        k += 1
     
    plot_error_partan(criterion_list, fx_list, d_list, mu)   
    print(len(criterion_list) + 1)
       
if __name__ == "__main__":
    #Gradient_Descent_Method(1e-5)
    Partan_Method(1e-4)