# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:42:41 2020

@author: jpatr_000
"""
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Linear regression model with L2 regualrization
# =============================================================================

class Model_Linear_L2:

        def __int__(self):
            pass

# Returns cost | Input: Training data - X , Training Labels - y , Weights - theta
# Penalization factor - lumbda
            

        def calc_cost(self, X, y, theta ,lumbda):
            m = len(y)
            cost  = 0.5/m * (np.sum(np.dot(X,theta) - y))**2 + (0.5 * lumbda / m) * np.dot(theta, theta.T)
            return cost
        
        
        
# Returns the derivatives of loss with respect to weights 
# Input: Training data - X , Training Labels - y , Weights - theta
# Penalization factor - lumbda
        
        
        def calc_grad(self, X, y, theta ,lumbda):
            m = len(y)
            theta_grad = (1/m) * np.dot ( X.T , ( np.dot( X , theta ) - y )) + lumbda * theta 
            return theta_grad
        


# Initializes the parameter with random values (can be replaced 0s as well)
        
        def initialize_param(self , num_param):
            return 0.01 * np.random.randn(num_param)


# It plots the cost decay from stored costs in model 
        
        def plot_cost(self , cost):
            plt.title('cost')
            plt.xlabel('iterations')
            plt.ylabel('cost')
            plt.plot(cost)
            plt.show()
            
        
# It invokes calc_cost() , calc_grad() and updated the weights after adjusting 
# gradients by learning rate and returns the final paramenters/ weights at the 
# end of all iterations along with cost comuted for each iteration            
        
        
        def gradient_descentt(self , X , y , iteration = 100 , lumbda = 0.03, learning_rate = 0.01 ):
            cost = []
            theta      = self.initialize_param(num_param = X.shape[1])
            
            for i in range(iteration):       
                if np.remainder(i, 1000) == 0:
                    learning_rate = 0.8 * learning_rate
                
                theta_grad  = self.calc_grad(X, y, theta , lumbda)
                theta       = theta - learning_rate * theta_grad 
                rmse = self.calc_cost(X, y, theta ,lumbda)
                cost.append(rmse)
                print("iteration   " + str(i) + "  of   " + str(iteration) + 
                      "   Cost is   " + str(rmse) + " & learning rate is :" + str(learning_rate))
            
            self.plot_cost(cost)    
            return {'param' : theta} 
    



# X = np.random.randn(1000).reshape(200,5)
# w = np.arange(5) + 1
# y = np.dot(X,w)
# linL2 = Model_Linear_L2()
# linL2.gradient_descentt(X , y , 1000, 0.01 ,  0.01)


