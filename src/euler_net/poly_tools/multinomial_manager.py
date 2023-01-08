from sympy import binomial, factorial
from sympy.ntheory.multinomial import multinomial_coefficients_iterator
import numpy as np


def nb_binomial_coef(d, k, sum=True):
    """
    Number of binomial coefficient for d and k fixed
    If sum is true then sum the number of polynomial coefficient 
    for d fixed and all index from 0 to k
    """
    if not sum:
        return round(binomial(k+d-1, d-1))
    else:
        return round(binomial(k+d, d))


def multi_index_factorial(coefs):
    res = 1
    for c in coefs:
        res *= factorial(c)
    return round(res)


class MultinomialManager():
    def __init__(self, d, k):
        self.d = d
        self.k = k
        # For k and d fixed the total number of basis function (k+d,d) 
        self.id_to_binom = [0]*nb_binomial_coef(d, k)
        # For k and d fixed the number of mutlinomial coefficient is (k+d-1,d-1) 
        # (we consider k+1 (same as before) here and k+2 for the rows (add one row for order 0))
        self.binom_to_id = np.zeros((k+1, nb_binomial_coef(d, k, sum=False)), dtype=int)
        self.binom_to_id.fill(-1)
        self.init_id_to_binomial()

    def init_id_to_binomial(self):
        d = self.d
        nb_previous_id = 0
        for i in range(self.k+1):
            it = multinomial_coefficients_iterator(d, i)
            for multi_coef in it:
                #for j in range(i+1):
                coef = multi_coef[0]
                id = nb_previous_id + round(coef[1])
                #self.binom_to_id[round(coef[0]), round(coef[1])] = id
                self.binom_to_id[coef] = id
                self.id_to_binom[id] = coef#(round(coef[0]), round(coef[1]))
            nb_previous_id = nb_binomial_coef(d, i)

    def binomial_to_id(self, p, l):
        """
        Return the index associated with x^p y^l
        ex: x^0 y^0 -> 1, x^1 y^0 -> 2, x^0 y^1 -> 3
        x^2 y^0 -> 4, x^1 y^1 -> 5, x^0 y^2 -> 6...
        """
        return self.binom_to_id[p, l]
        #return round(binomial(p+l+self.d, self.d) - 1 - p)

    def id_to_binomial(self, id):
        """
        Return the inverse of the function binomial_coef_to_index i.e. the index p, l associated with id
        """
        return self.id_to_binom[id]
    