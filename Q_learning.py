# -*- coding: utf-8 -*-
"""
Robust Q Learning
"""

import numpy as np
from tqdm import tqdm 
from scipy.optimize import minimize
import copy

def robust_q_learning(X,
               A,
               r,
               c,
               P_0, # Simulation of next state in dependence of x and a
               p_0,
               epsilon,
               alpha,
               x_0, 
               eps_greedy = 0.05,
               Nr_iter = 1000,
               q =1,
               gamma_t_tilde = lambda t: 1/(t+1),
               time_series = False,
               T = None,
               Q_0 = None):
    """
    

    Parameters
    ----------
    X : numpy.ndarray
        A list or numpy array containing all states
    A : numpy.ndarray
        A list or numpy array containing all actions
    r : function
        Reward function r(x,a,y) depending on state-action-state.
    c : function
        Function c(x,y) depending on state and state for the lambda c transform.
    P_0 : function
        fucntion P_0(x,a) that creates a new random variabe in dependence of state and action
    p_0 : function
        fucntion p_0(k,x,a) that determines the density at k given a state action pair (x,a)
    epsilon : float
        For the determination of the radius of the Wasserstein ball.
    alpha : float
        Discounting rate.
    x_0 : numpy.ndarray
        the initial state.
    eps_greedy : float, optional
        Parameter for the epsilon greedy policy. The default is 0.05.
    Nr_iter : int, optional
        Number of Iterations. The default is 1000.
    q : int, optional
        powert of the Wasserstein ball. The default is 1.
    gamma_t_tilde : function, optional
        learning rate. The default is lambda t: 1/(t+1).
    time_series : boolean, optional
        Whether we consider the time series setting or not. The default is False.
    T : numpy.ndarray, optional
        The smaller space for the time series setting. The default is None.
    Q_0 : matrix, optional
        Initial value for the Q-value matrix. The default is None.

    Returns
    -------
    matrix
        The Q value matrix.

    """
    rng = np.random.default_rng()
    #Initialize Q_0
    Q = np.zeros([len(X),len(A)])
    if Q_0 is not None:
        Q = Q_0
    # initialize a counter for the visits
    Visits = np.zeros([len(X),len(A)])
    # Bring A and X into the right format
    if np.ndim(A)>1:
        A_list = A
    else:
        A_list = np.array([[a] for a in A])
    if np.ndim(X)>1:
        X_list = X
    else:
        X_list = np.array([[x] for x in X])
    # Functions to determine the index of an action/state in A or X
    def a_index(a):
        return np.flatnonzero((a==A_list).all(1))[0]
    def x_index(x):
        return np.flatnonzero((x==X_list).all(1))[0]
    # Define the f_t function
    def f(t,x,a,y):
        return r(x,a,y)+alpha*np.max(Q[x_index(y),:])
    # Define the lambda c transform (already taking into account that we consider -f_t 
    def lambda_c_transform(t,x,a,y,lam):
        return np.max([-f(t,x,a,z)-lam*c(z,y) for z in X])
    # Define the epsilon greedy policy
    def a_t(t,y):
        #eps_bound = 1-(t/Nr_iter)*(eps_greedy)
        eps_bound = eps_greedy
        unif = np.random.uniform(0)
        return (unif>eps_bound)*A[np.argmax(Q[x_index(y),:])]+(unif<=eps_bound)*rng.choice(A)
    # The Expectation that is optimized w.r.t.lambda
    def expected_value_to_optimize(t,x,a,lam):
        return np.sum([(-lambda_c_transform(t,x,a,k,lam))*p_0(k,x,a) for k in X])-(epsilon**q)*lam
    # The Expectation that is optimized w.r.t.lambda in the time series case    
    def expected_value_to_optimize_ts(t,x,a,lam):
        return np.sum([(-lambda_c_transform(t,x,a,np.concatenate([x[:-1],[k]]),lam))*p_0(np.concatenate([x[:-1],[k]]),x,a) for k in T])-(epsilon**q)*lam
        
    # Set initial value        
    X_0 = x_0
    lam_0 = 1
    #lam_list = []
    cons = []
    l = {'type': 'ineq',
         'fun': lambda x: x - 0}
    u = {'type': 'ineq',
         'fun': lambda x: 1000 - x}
    cons.append(l)
    cons.append(u)
    # List of differences of Q Matrices 
    
    for t in tqdm(range(Nr_iter)):
        X_1 = P_0(X_0,a_t(t,X_0))
        Q_old = copy.deepcopy(Q)
        x,a = X_0, a_t(t,X_0)
        x_ind, a_ind = x_index(x),a_index(a)
        # Choose the maximal lambda
        if time_series:
            lam_t = minimize(lambda lam: -expected_value_to_optimize_ts(t,x,a,lam), 
                             x0 = lam_0,
                             bounds = [(0,None)]).x         
        else:
            lam_t = minimize(lambda lam: -expected_value_to_optimize(t,x,a,lam), 
                             x0 = lam_0,
                             bounds = [(0,None)]).x
        lam_0 = lam_t
        #lam_list.append(lam_t)
        # Do the update of Q        
        Q[x_ind, a_ind] = Q_old[x_ind, a_ind]+gamma_t_tilde(Visits[x_ind, a_ind])*(-lambda_c_transform(t,x,a,X_1,lam_t)-(epsilon**q)*lam_t-Q_old[x_ind, a_ind])
        Visits[x_ind, a_ind]+=1
        X_0 = X_1
    return Q

# Classical Q learning #

def q_learning(X,
               A,
               r,
               P_0, # Simulation of next state in dependence of x and a
               alpha,
               x_0, 
               eps_greedy = 0.05,
               Nr_iter = 1000,
               gamma_t_tilde = lambda t: 1/(t+1),
               Q_0 = None):
    """
    Parameters
    ----------
    X : numpy.ndarray
        A list or numpy array containing all states
    A : numpy.ndarray
        A list or numpy array containing all actions
    r : function
        Reward function r(x,a,y) depending on state-action-state.
    P_0 : function
        fucntion P_0(x,a) that creates a new random variabe in dependence of state and action
    alpha : float
        Discounting rate.
    x_0 : numpy.ndarray
        the initial state.
    eps_greedy : float, optional
        Parameter for the epsilon greedy policy. The default is 0.05.
    Nr_iter : int, optional
        Number of Iterations. The default is 1000.
    gamma_t_tilde : function, optional
        learning rate. The default is lambda t: 1/(t+1).
    Q_0 : matrix, optional
        Initial value for the Q-value matrix. The default is None.

    Returns
    -------
    matrix
        The Q value matrix.
    """
    rng = np.random.default_rng()    
    #Initialize Q_0
    Q = np.zeros([len(X),len(A)])
    if Q_0 is not None:
        Q = Q_0
    Visits = np.zeros([len(X),len(A)])
    if np.ndim(A)>1:
        A_list = A
    else:
        A_list = np.array([[a] for a in A])
    if np.ndim(X)>1:
        X_list = X
    else:
        X_list = np.array([[x] for x in X])

    def a_index(a):
        return np.flatnonzero((a==A_list).all(1))[0]
    def x_index(x):
        return np.flatnonzero((x==X_list).all(1))[0]
    
    # Define the f function
    def f(t,x,a,y):
        return r(x,a,y)+alpha*np.max(Q[x_index(y),:])
  
    def a_t(t,y):
        #eps_bound = 1-(t/Nr_iter)*(eps_greedy)
        eps_bound = eps_greedy
        unif = np.random.uniform(0)
        return (unif>eps_bound)*A[np.argmax(Q[x_index(y),:])]+(unif<=eps_bound)*rng.choice(A)
        
    
    # Set initial value        
    X_0 = x_0
    # List of differences of Q Matrices 
    for t in tqdm(range(Nr_iter)):
        X_1 = P_0(X_0,a_t(t,X_0))
        Q_old = copy.deepcopy(Q)
        x,a = X_0, a_t(t,X_0)
        x_ind, a_ind = x_index(x),a_index(a)
        # Do the update of Q        
        Q[x_ind, a_ind] = Q_old[x_ind, a_ind]+gamma_t_tilde(Visits[x_ind, a_ind])*(f(t,x,a,X_1)-Q_old[x_ind, a_ind])
        Visits[x_ind, a_ind]+=1
        X_0 = X_1
    return Q
        
    