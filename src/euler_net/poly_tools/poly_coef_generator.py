import numpy as np
from ...case import Case
from sympy.ntheory.multinomial import multinomial_coefficients_iterator
from .multinomial_manager import nb_binomial_coef, multi_index_factorial



class PolyCoefGenerator():
    def __init__(self, d, n, poly_case=Case.classic_poly, binom_normalization=False, init_order_first=False):
        self.n = n
        self.d = d
        self.init_order_first = init_order_first
        if poly_case == Case.classic_poly:
            self.nb_basis = d * nb_binomial_coef(d, n)
        elif poly_case == Case.incompressible_poly:
            self.nb_basis = (d-1) * nb_binomial_coef(d, n) + nb_binomial_coef(d-1, n)
        elif poly_case == Case.classic_scalar_poly:
            self.nb_basis = nb_binomial_coef(d, n)
        if poly_case != Case.classic_scalar_poly:
            # The value of the first index degrees[i,:,:]  represents the degrees of all the variables x_k 
            # for a given component i
            # The value of the second index degrees[:,i,:] represents the degrees of the variable x_i 
            # in all the basis functions
            # The value of the last index degrees[:,:,i] represents the degrees of all the variables x_k 
            # for a given basis function f_i
            self.basis_degrees = np.zeros((d, d, self.nb_basis), dtype=int)
            self.basis_coef = np.zeros((d, self.nb_basis))
        else:
            self.basis_degrees = np.zeros((1, d, self.nb_basis), dtype=int)
            self.basis_coef = np.zeros((1, self.nb_basis))

        #self.order_last_id = np.zeros(n+1, dtype=np.int)
        self.binom_normalization = binom_normalization
        if poly_case == Case.classic_poly:
            self.generate_classic()
        elif poly_case == Case.incompressible_poly:
            self.generate_incompressible()
        elif poly_case == Case.classic_scalar_poly:    
            self.generate_classic_scalar()
        

    def generate_classic_scalar(self):
        n = self.n
        d = self.d
        id_basis = 0
        #count_basis_fix_order = 0

        for p in range(n+1):
            #count_basis_fix_order = 0                        
            it = multinomial_coefficients_iterator(d, p)
            for multi_coef in it:
                derivatives_coef = multi_coef[0]
                binomial_coef = 1.
                self.add_coefficient(binomial_coef, derivatives_coef, 0, id_basis)
                id_basis += 1
                #count_basis_fix_order += 1
            #self.order_last_id[p] = count_basis_fix_order

    def generate_classic(self):
        n = self.n
        d = self.d
        id_basis = 0        

        if not self.init_order_first:
            for i in range(d):
                for p in range(n+1):                              
                    id_basis = self.generate_classic_monoms(i,p, id_basis)
        else:
            for p in range(n+1):
                for i in range(d):                
                    id_basis = self.generate_classic_monoms(i,p, id_basis)
                """
                it = multinomial_coefficients_iterator(d, p)
                for multi_coef in it:
                    derivatives_coef = multi_coef[0]
                    if self.binom_normalization:
                        binomial_coef = 1./multi_index_factorial(derivatives_coef)#multi_coef[1] / factorial_p
                    else:
                        binomial_coef = 1.
                    self.add_coefficient(binomial_coef, derivatives_coef, i, id_basis)
                    id_basis += 1
                    #count_basis_fix_order += 1
            #self.order_last_id[p] = count_basis_fix_order
            """

    def generate_classic_monoms(self, i, p, id_basis):
        it = multinomial_coefficients_iterator(self.d, p)
        for multi_coef in it:
            derivatives_coef = multi_coef[0]
            if self.binom_normalization:
                binomial_coef = 1./multi_index_factorial(derivatives_coef)#multi_coef[1] / factorial_p
            else:
                binomial_coef = 1.
            self.add_coefficient(binomial_coef, derivatives_coef, i, id_basis)
            id_basis += 1
        return id_basis

    def generate_incompressible(self):
        n = self.n
        d = self.d
        id_basis = 0
        if not self.init_order_first:        
            for i in range(d):
                for p in range(n+1):
                    id_basis = self.generate_incompressible_monoms(i, p, id_basis)
        else:
            for p in range(n+1):
                for i in range(d):                
                    id_basis = self.generate_incompressible_monoms(i, p, id_basis)                              
    
    def generate_incompressible_monoms(self, i, p, id_basis):                    
        it = multinomial_coefficients_iterator(self.d, p)
        for multi_coef in it:
            derivatives_coef = list(multi_coef[0])
            if self.binom_normalization:                        
                binomial_coef = 1./multi_index_factorial(derivatives_coef)
            if i==0 and derivatives_coef[0] == 0:
                if not self.binom_normalization:
                    binomial_coef = derivatives_coef[1]+1
                    # binomial_coef = 1.
                self.add_coefficient(binomial_coef, derivatives_coef, i, id_basis)
                id_basis += 1                
            elif i > 0:
                if not self.binom_normalization:
                    binomial_coef = derivatives_coef[0]+1                        
                    #if derivatives_coef[i] == 0:
                    #    binomial_coef = 1.
                    #else:
                    #    binomial_coef = 1. / derivatives_coef[i]            
                self.add_coefficient(binomial_coef, derivatives_coef, i, id_basis)
                if derivatives_coef[i] > 0:
                    derivatives_coef[0] += 1
                    derivatives_coef[i] -= 1
                    if self.binom_normalization:                                
                        binomial_coef = -1./multi_index_factorial(derivatives_coef)
                    else:
                        binomial_coef = -(derivatives_coef[i]+1)
                        # binomial_coef = -1. / derivatives_coef[0]
                    self.add_coefficient(binomial_coef, derivatives_coef, 0, id_basis)    
                id_basis += 1
        return id_basis
                
    
    def add_coefficient(self, scalar_coef, derivatives_coef, id_component, id_basis):
        self.basis_coef[id_component, id_basis] = scalar_coef
        id_derivative = 0
        for i in derivatives_coef:
            self.basis_degrees[id_component, id_derivative, id_basis] += round(i)
            id_derivative += 1