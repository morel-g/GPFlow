
import numpy as np
from case import Case
from sympy import binomial, factorial
from sympy.ntheory.multinomial import multinomial_coefficients_iterator
from poly_tools.multinomial_manager import nb_binomial_coef, multi_index_factorial


class PolyCoefGenerator():
    def __init__(self, d, n, poly_case=Case.classic_poly, binom_normalization=True):
        self.n = n
        self.d = d
        if poly_case == Case.classic_poly:
            self.nb_basis = d * nb_binomial_coef(d, n)
        elif Case.incompressible_poly:
            self.nb_basis = (d-1) * nb_binomial_coef(d, n) + nb_binomial_coef(d-1, n)
        # The value of the first index degrees[i,:,:]  represents the degrees of all the variables x_k 
        # for a given component i
        # The value of the second index degrees[:,i,:] represents the degrees of the variable x_i 
        # in all the basis functions
        # The value of the last index degrees[:,:,i] represents the degrees of all the variables x_k 
        # for a given basis function f_i
        self.basis_degrees = np.zeros((d, d, self.nb_basis), dtype=int)
        self.basis_coef = np.zeros((d, self.nb_basis))
        self.order_last_id = np.zeros(n+1, dtype=np.int)

        if poly_case == Case.classic_poly:
            self.generate_classic()
        elif Case.incompressible_poly:
            self.generate_incompressible()


    def generate_classic(self):
        n = self.n
        d = self.d
        id_basis = 0
        count_basis_fix_order = 0

        for p in range(n+1):
            count_basis_fix_order = 0            
            for i in range(d):
                it = multinomial_coefficients_iterator(d, p)
                for multi_coef in it:
                    derivatives_coef = multi_coef[0]
                    binomial_coef = 1./multi_index_factorial(derivatives_coef)#multi_coef[1] / factorial_p
                    self.add_coefficient(binomial_coef, derivatives_coef, i, id_basis)
                    id_basis += 1
                    count_basis_fix_order += 1
            self.order_last_id[p] = count_basis_fix_order

    def generate_incompressible(self):
        n = self.n
        d = self.d
        id_basis = 0
        count_basis_fix_order = 0
        for p in range(n+1):
            count_basis_fix_order = 0            
            for i in range(d):
                it = multinomial_coefficients_iterator(d, p)
                for multi_coef in it:
                    derivatives_coef = list(multi_coef[0])
                    #binomial_coef = 1./multi_index_factorial(derivatives_coef)
                    if i==0 and derivatives_coef[0] == 0:
                        binomial_coef = derivatives_coef[1]+1
                        self.add_coefficient(binomial_coef, derivatives_coef, i, id_basis)
                        id_basis += 1
                        count_basis_fix_order += 1
                    elif i > 0:
                        binomial_coef = derivatives_coef[0]+1
                        self.add_coefficient(binomial_coef, derivatives_coef, i, id_basis)
                        if derivatives_coef[i] > 0:
                            derivatives_coef[0] += 1
                            derivatives_coef[i] -= 1
                            # binomial_coef = -1./multi_index_factorial(derivatives_coef)
                            binomial_coef = -(derivatives_coef[1]+1)
                            self.add_coefficient(binomial_coef, derivatives_coef, 0, id_basis)    
                        id_basis += 1
                        count_basis_fix_order += 1
            self.order_last_id[p] = count_basis_fix_order


    def add_coefficient(self, scalar_coef, derivatives_coef, id_component, id_basis):
        self.basis_coef[id_component, id_basis] = scalar_coef
        id_derivative = 0
        for i in derivatives_coef:
            self.basis_degrees[id_component, id_derivative, id_basis] += round(i)
            id_derivative += 1