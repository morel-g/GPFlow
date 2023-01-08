import numpy as np
from ...case import Case
from ..poly_tools.multinomial_manager import MultinomialManager, nb_binomial_coef
import torch

from ...precision import torch_float_precision


class CircleBoundary():
    """
       Impose linear relations on the coefficients to satisfy $q(x) := p(x) \cdot n(x) = 0$ on a circle
    of radius $r$. To do so we make the expansion 
    \begin{equation}
        \label{eq:trigo-expansion}
        q(r, \theta) = \alpha_0^0 + \sum_{k=0}^n \sum_{l=0}^k r^k \alpha_k^l \cos^{k-l} \theta \sin^l \theta.
    \end{equation}
    Then we use the equality $\sin^2\theta = 1 - \cos^2\theta$ to write $q$ under the form
    \begin{equation}
        \label{eq:reduce-trigo-expansion}
        q(r, \theta) = \beta_0^0 + \sum_{k=0}^n \beta_k^k \cos^k \theta + \beta_k^{k-1} \cos^{k-1}\theta \sin\theta.
    \end{equation}
    And one has $\beta_k^k = \beta_k^{k-1} = 0$, for all $k$.
    
    We use a recurrent matrix $R$ to track the dependence in the basis functions in the parameters $\beta_k^p$.
    More precisely the indexes of the rows of $R$ i.e.$ R[i,:]$ denotes the index of the trigonometric polynomial ex:
    \begin{equation*}
        1 \rightarrow 0, \quad \cos\theta \rightarrow 1, \quad \sin\theta \rightarrow 2, \quad \cos^2\theta \rightarrow 3, \quad \cos\theta\sin\theta \rightarrow 4, \quad \sin^2\theta \rightarrow 5...
    \end{equation*}
    The indexes of the colum of $R$ i.e. $R[:,j]$ denotes the index of the basis functions for example:
    \begin{equation*}
        (1, 0) \rightarrow 0, \quad (0, 1) \rightarrow 1, \quad (y, 0) \rightarrow 2, \quad (x, 0) \rightarrow 3, \quad (0, y) \rightarrow 4, \quad (0, x) \rightarrow 5 ...
    \end{equation*}
    Once the matrix $R$ is completed with the recurrence relations we impose the relations by removing 
    the indexes corresponding to the basis functions 
    \begin{equation}
        \label{eq:remove-basis-shape}
        (\cos^k\theta, 0) \quad \text{or} \quad (0, \cos^k \theta) ,
     \end{equation}
    and the special case $(0, \sin\theta)$.
    Let's say that the basis function $(\cos^k\theta, 0)$ we want to remove has id $j$ and $\cos^{k+1}$ has id $i$ 
    in the expansion \eqref{eq:reduce-trigo-expansion}. We set
    $R[i, :] = -R[i,:] / R[i,j]$ then we remove the column $j$ and the coefficient $\alpha_j$ is given 
    by the other coefficients $\alpha$ as $\alpha_j = R[i,:] \alpha$.

    Remarque (special case): We also need to set $\beta_0^0=0$ but this coefficient is not associated 
    with a function under the form \eqref{eq:remove-basis-shape}. So we impose this relation on the 
    coefficient associated with $(0, \sin\theta)$. Because this coefficient is zeros we DO NOT make the 
    recurrence for this term. 
    Moreover we do not set this term to zero for incompressible polynomials because this term 
    is a linear combination of some of the other terms which are set to zeros.
    """
    def __init__(self, d, n, basis_degrees, basis_coef, radius=1,
                 poly_case=Case.classic_poly, debug=False):
        self.n = n
        self.d = d
        self.basis_degrees = basis_degrees
        self.basis_coef = basis_coef
        self.radius = radius
        self.poly_case = poly_case
        # Take n+1 since we want ceofficient for $p \cdot n$
        self.multinomial = MultinomialManager(d, n+1)
        #print("Nb basis out = ",basis_degrees.shape[2])
        #print("k+d, d = ", 2* binomial(d+n,d))
        self.nb_basis_out = basis_degrees.shape[2]
        if poly_case==Case.classic_poly:
            self.nb_basis_in = self.nb_basis_out-(2*(n+1)+1)
        else:
            self.nb_basis_in = self.nb_basis_out-(2*(n+1)+1) +1 

        # Recurrence matrix of the polynomial $p^\cdot n$
        self.R = np.zeros((nb_binomial_coef(d, n+1), self.nb_basis_out))
        self.boundary_matrix = np.zeros((self.nb_basis_out, self.nb_basis_in))
        self.global_to_local_id = np.zeros(self.nb_basis_out, dtype=int)
        self.local_to_global_id = np.zeros(self.nb_basis_in, dtype=int)
        
        if not debug:
            self.init_recurrent_matrix()
            self.fill_recurrent_matrix()
            self.remove_recurrent_ids()
            self.fill_boundary_matrix()

    def apply_boundary_conditions(self, alpha):        
        return torch.matmul(self.boundary_matrix, alpha) 

    def fill_boundary_matrix(self):
        for basis_id in range(self.boundary_matrix.shape[0]):
            local_id = self.global_to_local_id[basis_id]
            use_R, trigo_poly_id = self.id_to_remove(basis_id)
            
            if use_R:
                self.boundary_matrix[basis_id, :] = self.R[trigo_poly_id, :]
            else:
                self.boundary_matrix[basis_id, local_id] = 1
                self.local_to_global_id[local_id] = basis_id

        self.boundary_matrix = torch.tensor(self.boundary_matrix, dtype=torch_float_precision)

    def id_to_remove(self, basis_id):
        degrees = self.basis_degrees
        coef = self.basis_coef    
        remove_id = False
        trigo_poly_id = -1
        # If we do have x's degree but no y's degree
        # for the first component
        if (degrees[0, 1, basis_id] == 0 and coef[0, basis_id] != 0):
            trigo_poly_id = self.multinomial.binomial_to_id(
                degrees[0, 0, basis_id]+1, 0)
            remove_id = True
        # If we do have x's degree but no y's degree
        # for the second component
        elif (degrees[1, 1, basis_id] == 0 and coef[1, basis_id] != 0):
            trigo_poly_id = self.multinomial.binomial_to_id(
                degrees[1, 0, basis_id], 1)
            remove_id = True
        elif (self.poly_case == Case.classic_poly):
            if (degrees[1, 1, basis_id] == 1 and degrees[1, 0, basis_id] == 0
                and coef[1, basis_id] != 0):
                trigo_poly_id = self.multinomial.binomial_to_id(0, 2)
                remove_id = True
        else:
            pass
            #if (degrees[1, 1, basis_id] == 3 and degrees[0, 1, basis_id] == 0
            #    and coef[1, basis_id] != 0):
            #    trigo_poly_id = self.multinomial.binomial_to_id(0, 2)
            #    remove_id = True       

        return remove_id, trigo_poly_id

    def remove_recurrent_ids(self):
        R = self.R
        col_to_remove = np.ones(self.basis_degrees.shape[-1], dtype=int)             

        for basis_id in range(self.basis_degrees.shape[-1]):
            remove_id, trigo_poly_id = self.id_to_remove(basis_id)

            if remove_id:
                R[trigo_poly_id, :] = (- R[trigo_poly_id, :] /
                                       R[trigo_poly_id, basis_id])

                # Only necessary for incompressible poly
                for id in range(R.shape[0]-1,-1,-1):
                    if id != trigo_poly_id:
                        if R[id, basis_id] != 0.:
                            R[id, :] = R[id, :] + R[id, basis_id] * R[trigo_poly_id, :]

                col_to_remove[basis_id] = 0

        self.global_to_local_id = np.cumsum(col_to_remove)-1
        self.R = R[:, col_to_remove==1]

    def fill_recurrent_matrix(self):
        for n in range(self.n+1, -1, -1):
            for l in range(n, 1, -1):
                id = self.multinomial.binomial_to_id(n-l, l)
                id2 = self.multinomial.binomial_to_id(n-l+2, l-2)
                if (n!=2 or l!=2):
                    self.R[id2, :] += -self.R[id, :]
                id2 = self.multinomial.binomial_to_id(n-l, l-2)
                self.R[id2, :] += self.R[id, :]

    def init_recurrent_matrix(self):
        if self.d != 2:
            raise RuntimeError("Reccurrent matrix only works with d=2.")

        r = self.radius
        
        for basis_id in range(self.basis_degrees.shape[-1]):
            for x_i in range(self.basis_degrees.shape[0]):
                degrees_cos = self.basis_degrees[x_i, 0, basis_id]
                degrees_sin = self.basis_degrees[x_i, 1, basis_id]

                if x_i == 0:
                    # Polynomial $p \cdot n$ gains one degree for the cos
                    # when considering the first component
                    degrees_cos += 1
                else:
                    # Polynomial $p \cdot n$ gains one degree for the sin 
                    # when considering the second component
                    degrees_sin += 1
                index = self.multinomial.binomial_to_id(
                    degrees_cos, degrees_sin)
                self.R[index, basis_id] += (
                    r**(degrees_cos+degrees_sin-1)
                    * self.basis_coef[x_i, basis_id])
