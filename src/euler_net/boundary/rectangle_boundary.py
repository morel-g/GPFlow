import numpy as np
from ...case import Case
from ..poly_tools.multinomial_manager import nb_binomial_coef
from ...precision import torch_float_precision
from sympy.ntheory.multinomial import multinomial_coefficients_iterator
import torch


class RectangleBoundary():

    def __init__(self, d, n, basis_degrees, basis_coef, bounds=(0,1,0,1),
                 poly_case=Case.classic_poly, debug=False, pseudo_inverse=True):
        self.n = n
        self.d = d
        self.basis_degrees = basis_degrees
        self.basis_coef = basis_coef
        self.bounds = bounds
        self.poly_case = poly_case
        # Take n+1 since we want ceofficient for $p \cdot n$
        # self.multinomial = MultinomialManager(d, n)
        self.nb_basis_out = basis_degrees.shape[2]
        self.nb_basis_in = self.nb_basis_out-(2*d*nb_binomial_coef(d-1, n, sum=True)-d)#2*nb_binomial_coef(d, n-2)#self.nb_basis_out-(4*n+4)
        if poly_case==Case.incompressible_poly:
            #self.nb_basis_in +=1
            self.nb_basis_in -= 0
        #self.nb_basis_in +=-n-2
        #print("nb_basis_in = ", self.nb_basis_in)
        # The rows of the R matrix is composed as follow
        # R[0,:,:] -> p_1(x,y_min), R[1,:,:] -> p_1(x,y_max)
        # R[2,:,:] -> p_2(x_min, y), R[3,:,:] -> p_2(x_max, y)
        # Add +d ?
        self.constraint_matrix = np.zeros((2*d*nb_binomial_coef(d-1, n, sum=True), self.nb_basis_out))
        self.Id = np.eye(d, d, dtype=int) 
        self.R = torch.eye(self.nb_basis_out, self.nb_basis_out, dtype=torch.float64)
        if not pseudo_inverse:
            self.boundary_matrix = torch.zeros((self.nb_basis_out, self.nb_basis_in))#np.eye(self.nb_basis_out, dtype=int)
        #self.removes_id = {}#np.ones((4, n+1), dtype=int)*(-1) #[[], [], [], []]
        if not pseudo_inverse:
            self.global_to_local_id = np.zeros(self.nb_basis_out, dtype=int)
            self.local_to_global_id = np.zeros(self.nb_basis_in, dtype=int)
        self.col_to_remove = np.ones(self.basis_degrees.shape[-1], dtype=int)

        # List composed of $l$ lists for $1 \leq l \leq d$ corresponding to the index 
        # for the constraint of $alpha^l_{\mathring{\imath}_l}$ 
        self.M1 = np.zeros([d] + (d-1)*[n+1], dtype=int)
        # List composed of $l$ lists for $1 \leq l \leq d$ corresponding to the index 
        # for the constraint of $alpha^l_{\mathring{\imath}_l + e_l}$ 
        self.M2 = np.zeros([d] + (d-1)*[n+1], dtype=int)
        self.basis_non_zeros_component = np.zeros(self.basis_degrees.shape[-1], dtype=int)-1
        self.alpha1_32 = -1
        self.alpha2_03 = -1
        self.count_remove = 0
        #self.multinomial_ids = np.zeros((d, nb_binomial_coef(d-1, n, sum=True)), dtype=int)
        self.multinomial_ids = np.zeros(([d]+[n+1]*(d-1)), dtype=int)

        if not pseudo_inverse:
            self.init_R()
            self.compute_boundary_matrix()
        else:
            self.fill_constraint_matrix()            

    def apply_boundary_conditions(self, alpha):        
        return torch.matmul(self.boundary_matrix, alpha) 

    def init_multinomial_id(self):
        eps = 1e-6
        nb_previous_id = 0
        for l in range(self.d):
            for i in range(self.n+1):
                it = multinomial_coefficients_iterator(self.d-1, i)
                for multi_coef in it:
                    #for j in range(i+1):
                    coef = multi_coef[0]
                    id = nb_previous_id 
                    #self.binom_to_id[round(coef[0]), round(coef[1])] = id                    
                    self.multinomial_ids[(l,)+ coef] = id                    
                    nb_previous_id += 1
            
        
    def fill_constraint_matrix(self):
        self.init_multinomial_id()
        eps = 1e-6
        max_id_row = 0
        for basis_id in range(self.basis_degrees.shape[-1]):
            coef = self.basis_coef[:, basis_id]
            for l in range(coef.shape[0]):
                if abs(coef[l]) > eps:
                    a_l, b_l = self.bounds[l][0], self.bounds[l][1]
                    deg = self.basis_degrees[:,:,basis_id]
                    coef = self.basis_coef[:,basis_id]
                    
                    id_row = 2*self.multinomial_ids[(l,) +tuple(deg[l,:][torch.arange(self.d)!=l])]
                    
                    """ 
                    if l==0:
                        id_row = 2*l*(self.n+1)+2*deg[l,1]
                    else:
                        id_row = 2*l*(self.n+1)+2*deg[l,0]
                    """
                    self.constraint_matrix[id_row, basis_id] = coef[l]*(a_l**deg[l,l])
                    self.constraint_matrix[id_row+1, basis_id] = coef[l]*(b_l**deg[l,l])
                    max_id_row = max(max_id_row, id_row+1)
        #print("max_id_row = ", max_id_row)
        #print("constraint matrix rows = ", self.constraint_matrix.shape[0])

        #pinv = np.linalg.pinv(self.constraint_matrix)
        #print(np.allclose(pinv, np.dot(pinv, np.dot(self.constraint_matrix, pinv))))
        #print(np.allclose(self.constraint_matrix, np.dot(self.constraint_matrix, np.dot(pinv, self.constraint_matrix))))

        #self.boundary_matrix2 = torch.tensor(np.eye(self.constraint_matrix.shape[1]) - np.dot(pinv, self.constraint_matrix))
        #u, s, vh = np.linalg.svd(self.boundary_matrix)
        #ids = np.where(abs(s) > 1e-10)
        u, s, vh = np.linalg.svd(self.constraint_matrix)
        s2 = np.concatenate((s, np.zeros(vh.shape[0]-s.shape[0])))
        null_space = np.compress(abs(s2) <= 1e-10, vh, axis=0)
        self.boundary_matrix =torch.transpose(torch.tensor(null_space, dtype=torch_float_precision),0,1)
        # print("shape boundary_matrix = ", self.boundary_matrix.shape)
        # nb basis functions in = self.basis_out - self.d*(2*nb_binomial_coef(self.n,seld.d-1)-(d-1))
        # But some bugs for example incompressible poly with d=2 n=3
        self.nb_basis_in = self.boundary_matrix.shape[1]
        # print("Top")
        


    def fill_id_to_remove(self, basis_id):
        remove_id = False
        id = -1, -1

        coef = self.basis_coef[:, basis_id]
        deg = self.basis_degrees[:, :, basis_id]
        d = self.d

        for l in range(coef.shape[0]):
            if coef[l] != 0.:
                if deg[l,l] == 0:
                    self.M1[l, deg[l, torch.arange(d)!=l]] = basis_id
                    self.count_remove += 1
                    if self.col_to_remove[basis_id] == 0:
                        RuntimeError("Error already filled")
                    self.col_to_remove[basis_id] = 0
                    
                elif deg[l,l] == 1 and not(
                    (l==0 and deg[l,1]==2)
                    or (l==1 and deg[l, 0]==0)
                ):
                    self.M2[l, deg[l, torch.arange(d)!=l]] = basis_id
                    self.count_remove += 1
                    if self.col_to_remove[basis_id] == 0:
                        RuntimeError("Error already filled")
                    self.col_to_remove[basis_id] = 0                    
                elif l==0 and deg[l,0]==3 and deg[l,1]==2:
                    self.alpha1_32 = basis_id
                    self.count_remove += 1
                    if self.col_to_remove[basis_id] == 0:
                        RuntimeError("Error already filled")
                    self.col_to_remove[basis_id] = 0
                elif l==1 and deg[l,0]==0 and deg[l,1]==3:
                    self.alpha2_03 = basis_id
                    self.count_remove += 1
                    if self.col_to_remove[basis_id] == 0:
                        RuntimeError("Error already filled")
                    self.col_to_remove[basis_id] = 0
                if not (l==0 and deg[l,0]>0):
                    self.basis_non_zeros_component[basis_id] = l

    def init_R(self):
        d = self.d
        n = self.n
        if self.d != 2:
            raise RuntimeError("Reccurrent matrix only works with d=2.")

        bounds = self.bounds

        R = self.R
        for basis_id in range(self.basis_degrees.shape[-1]):
            self.fill_id_to_remove(basis_id)

        if 2*d*nb_binomial_coef(d-1, n, sum=True)-d != self.count_remove:
            print("Error counting number of remove ids.")
            self.boundary_matrix = np.zeros((self.nb_basis_out, self.nb_basis_out-self.count_remove))

        for basis_id in range(self.basis_degrees.shape[-1]):
            coef = self.basis_coef[:, basis_id]
            deg = self.basis_degrees[:,:,basis_id]
            l = self.basis_non_zeros_component[basis_id]
            a_l, b_l = bounds[l][0], bounds[l][1]
            a_1, b_1 = bounds[0][0], bounds[0][1]
            # alpha^l_ij for j >= 2
            if l!=0:
                if deg[l, 0]!=0 and deg[l, l]>=2:  
                    # id row for alpha^l_{\mathring{\imath}_l + e_l}
                    id_row = self.M2[l, deg[l, torch.arange(d)!=l]]
                    deg_row_1 = self.basis_degrees[:, :, id_row]
                    coef_row_1 = self.basis_coef[:, id_row]
                    contrib = coef[l]*(a_l**deg[l, l]-b_l**deg[l, l])/((b_l-a_l)*coef_row_1[l])
                    R[id_row, basis_id] += contrib

                    # Contribution of alpha^2_{i1} for alpha^2_{i0}
                    id_row = self.M1[l, deg[l, torch.arange(d)!=l]]
                    coef_row_2 = self.basis_coef[:, id_row]                    
                    R[id_row, basis_id] += (-coef[l]*a_l**deg[l,l] - coef_row_1[l]*a_l*contrib) / coef_row_2[l]

                    # Contribution for alpha^2_{01}=alpha^1_{10}
                    id_row = self.M2[0, 0]
                    coef_row_3 = self.basis_coef[:, id_row]                    
                    R[id_row, basis_id] += coef_row_1[0]*contrib*(a_1**(deg_row_1[l,0]+1)-b_1**(deg_row_1[l,0]+1)) / ((b_1-a_1)*coef_row_3[0])
                
                    # Contribution of alpha^2_{01} for alpha^1_{00}                    
                    id_row = self.M1[0, 0]
                    coef_row_4 = self.basis_coef[:, id_row]
                    R[id_row, basis_id] += -(coef_row_3[0] * contrib * a_1) /coef_row_4[0]
                elif deg[l, l]>=1:
                    # Contribution for alpha^2_{03}
                    id_row = self.alpha2_03
                    coef_row_1 = self.basis_coef[:, id_row]
                    contrib = coef[l]*(a_l**deg[l, l]-b_l**deg[l, l])/((b_l**3-a_l**3)*coef_row_1[l])
                    R[id_row, basis_id] += contrib

                    # Contribution for alpha^2_{00}                    
                    #id_row = self.M1[l, deg[l, torch.arange(d)!=l]]
                    id_row = self.M1[l, 0]
                    coef_row_2 = self.basis_coef[:, id_row]
                    R[id_row, basis_id] += (-coef[l]*a_l**deg[l,l] - coef_row_1[l]*(a_l**3)*contrib) / coef_row_2[l]

                    # Contribution of alpha^2_{03} for alpha^2_{23}=alpha^1_{32}
                    id_row = self.alpha1_32
                    coef_row_3 = self.basis_coef[:, id_row]
                    contrib_2 = coef_row_1[0]*contrib*(a_1-b_1) / ((b_1**3-a_1**3)*coef_row_3[0])
                    R[id_row, basis_id] += contrib_2

                    #Contribution of alpha^2_{23} for alpha^1_{02}
                    id_row = self.M1[0, 2]
                    coef_row_4 = self.basis_coef[:, id_row]
                    R[id_row, basis_id] += - coef_row_3[0] * contrib_2 * a_1**3 /(coef_row_4[0])
            if l!=0:    
                #for l=1 -> i != 0
                if not (deg[l, 1]==0 or deg[l, 1]==1 or deg[l, 1]==3):
                    # Contribution for alpha^2_{0,j+1}=alpha^1_{1,j}
                    id_row = self.M2[0, (deg[l, :]-self.Id[:, l])[torch.arange(d)!=0]]
                    coef_row_1 = self.basis_coef[:, id_row]
                    contrib = coef[0] *(a_1**(deg[l, 0]+1)-b_1**(deg[l, 0]+1)) / (coef_row_1[0]*(b_1-a_1))
                    R[id_row, basis_id] += contrib

                    # Contribution of alpha^2_{0,j+1} for alpha^1_{0j}
                    # What if j==0
                    id_row = self.M1[0, (deg[l, :]-self.Id[:, l])[torch.arange(d)!=0]]
                    coef_row_2 = self.basis_coef[:, id_row]
                    R[id_row, basis_id] += (- coef[0] * a_1**(deg[l, 0]+1) - coef_row_1[0] * contrib * a_1) /coef_row_2[0]
                
                if deg[l, 1]==3 and deg[l, 0]!=2 and deg[l, 0]!=0:
                    # Second contribution alpha^2_{23}=alpha^1_{32}
                    id_row = self.alpha1_32
                    coef_row_1 = self.basis_coef[:, id_row]
                    R[id_row, basis_id] += coef[0]*(a_1**(deg[l,0]+1)-b_1**(deg[l,0]+1)) / ((b_1**3-a_1**3)*coef_row_1[0])

                if deg[l, 1]==3 and deg[l, 0]!=2:
                    # Second contribution for alpha^1_{02}
                    id_row = self.M1[0, 2]
                    coef_row_1 = self.basis_coef[:, id_row]
                    R[id_row, basis_id] += (- coef[0] * a_1**(deg[l, 0]+1)) /coef_row_1[0]
                
    def compute_boundary_matrix(self):
        # Divide by the coefs!
        self.global_to_local_id = np.cumsum(self.col_to_remove)-1
        self.boundary_matrix = self.R[:, self.col_to_remove==1]
    

    """
    def fill_boundary_matrix(self):
        print("nb basis in = ", self.nb_basis_in)
        print("boundary matrix shape = ", self.boundary_matrix.shape)
        print("R shape = ", self.R.shape)
        for basis_id in range(self.boundary_matrix.shape[0]):
            if not self.null_id(basis_id):
                local_id = self.global_to_local_id[basis_id]
                use_R, id = self.id_to_remove(basis_id)
            
                if use_R: 
                    self.boundary_matrix[basis_id, :] = self.R[id[0], id[1], :]
                else:
                    self.boundary_matrix[basis_id, local_id] = 1
                    self.local_to_global_id[local_id] = basis_id

        self.boundary_matrix = torch.tensor(self.boundary_matrix, dtype=torch_float_precision)
    """

    def id_to_remove(self, basis_id):
        remove_id = False
        id = -1, -1

        coef = self.basis_coef[:, basis_id]
        deg = self.basis_degrees[:, :, basis_id]
        x_deg_0 = self.basis_degrees[0, 0, basis_id]
        y_deg_0 = self.basis_degrees[1, 0, basis_id]
        x_deg_1 = self.basis_degrees[0, 1, basis_id]
        y_deg_1 = self.basis_degrees[1, 1, basis_id]

        if coef[0]!=0.:
            if (x_deg_0==1 or x_deg_0==2) and (y_deg_0==2 or y_deg_0==3):
                id1 = 0 if y_deg_0==2 else 1
                id = id1, x_deg_0
                remove_id =True
            elif (x_deg_0!=1 and x_deg_0!=2) and (y_deg_0==0 or y_deg_0==1):
                id1 = 0 if y_deg_0==0 else 1
                id = id1, x_deg_0
                remove_id =True                
        if coef[1]!=0.:
            if (y_deg_1==1 or y_deg_1==2) and (x_deg_1==0 or x_deg_1==1):
                id1 = 2 if x_deg_1==0 else 3
                id = id1, y_deg_1
                if remove_id==True:
                    print("ERROR remove_id already set to true")
                remove_id =True
            elif y_deg_1==self.n-2 and x_deg_1==1:
                id = 3, y_deg_1
                if remove_id==True:
                    print("ERROR remove_id already set to true")
                remove_id =True
            elif y_deg_1!=1 and y_deg_1!=2 and (x_deg_1==2 or x_deg_1==3):
                id1 =  2 if x_deg_1==2 else 3
                id = id1, y_deg_1
                if remove_id==True:
                    print("ERROR remove_id already set to true")                
                remove_id =True

        return remove_id, id

    def null_id(self, basis_id):
        x_deg_0 = self.basis_degrees[0, 0, basis_id]
        y_deg_0 = self.basis_degrees[1, 0, basis_id]
        x_deg_1 = self.basis_degrees[0, 1, basis_id]
        y_deg_1 = self.basis_degrees[1, 1, basis_id]
        n = self.n
        return x_deg_0 == n or x_deg_0== n-1 or y_deg_1==n or y_deg_1==n-1

                

"""
    def remove_recurrent_ids(self):
        pass

    def fill_R(self):
        for n in range(self.n+1, -1, -1):
            for l in range(n, 1, -1):
                id = self.multinomial.binomial_to_id(n-l, l)
                id2 = self.multinomial.binomial_to_id(n-l+2, l-2)
                if (n!=2 or l!=2):
                    self.R[id2, :] += -self.R[id, :]
                id2 = self.multinomial.binomial_to_id(n-l, l-2)
                self.R[id2, :] += self.R[id, :]


    def init_boundary_matrix(self):

        for i in self.removes_id.shape[0]:
            for j in self.removes_id.shape[1]:
                basis_id = self.removes_id[i, j]
                self.boundary_matrix[basis_id, :] = self.R[i, j, :]
"""
"""
            if x_deg_0 == 0:
                self.removes_id[0, y_deg_0] = basis_id
            if y_deg_0 != 0 and x_deg_0 == 1:
                self.removes_id[1, y_deg_0] = basis_id
            elif y_deg_0 == 0 and x_deg_0 == 3:
                self.removes_id[1, y_deg_0] = basis_id
            if y_deg_1 == 0:
                self.removes_id[2, x_deg_1] = basis_id
            if x_deg_1 != 0 and y_deg_1 == 2:
                self.removes_id[3, x_deg_1] = basis_id
            elif x_deg_1 == 0 and y_deg_1 == 1:
                self.removes_id[3, x_deg_1] = basis_id
"""
        



"""
    def fill_boundary_matrix(self):

        for basis_id in range(self.basis_degrees.shape[-1]):
            # print("basis_id = ", basis_id)
            # print(" value = ", self.basis_degrees[:,:,basis_id])
            # print(', coef = ', self.basis_coef[:,basis_id])
            for x_i in range(self.basis_degrees.shape[1]):
                x_degree = self.basis_degrees[0, x_i, basis_id]
                y_degree = self.basis_degrees[1, x_i, basis_id]

                coef = self.basis_coef[x_i, basis_id]
                if coef != 0:
                    self.add_coefficient(x_i, basis_id, x_degree, y_degree)
        
        self.boundary_matrix = torch.tensor(self.boundary_matrix, dtype=torch_precision)

    def add_coefficient(self, x_i, basis_id, x_degree, y_degree):
        x_min, x_max, y_min, y_max = self.bounds
        boundary_matrix = self.boundary_matrix
        n = self.n
        x_forbidden_id = ((0, n), (1, n-1), (0, n-1))
        y_forbidden_id = ((n, 0), (n-1, 1), (n-1, 0))
        print("x_deg = ", x_degree, "y_deg = ", y_degree, " x_i = ",x_i)
        if x_i == 0 and (x_degree, y_degree) not in x_forbidden_id:
            if x_degree > 1:
                #scaling = coef / (1 - x_min - x_max +x_min*x_max)
                local_id = self.binomial_to_local_id(x_i, x_degree-2, y_degree)
                if x_degree + y_degree-2 < n-1:
                    boundary_matrix[basis_id, local_id] += 1
            if x_degree > 0 and x_degree < n:
                local_id = self.binomial_to_local_id(x_i, x_degree-1, y_degree)
                if x_degree + y_degree-1 < n-1:
                    boundary_matrix[basis_id, local_id] += -x_min-x_max
            if x_degree < n-1:
                local_id = self.binomial_to_local_id(x_i, x_degree, y_degree)
                if x_degree + y_degree-2 < n-1:
                    boundary_matrix[basis_id, local_id] += x_min*x_max
        if x_i == 1 and (x_degree, y_degree) not in y_forbidden_id:
            if y_degree > 1:
                #scaling = coef / (1 - x_min - x_max +x_min*x_max)
                local_id = self.binomial_to_local_id(x_i, x_degree, y_degree-2)
                if x_degree + y_degree-2 < n-1:
                    boundary_matrix[basis_id, local_id] += 1
            if y_degree > 0 and y_degree < n:
                local_id = self.binomial_to_local_id(x_i, x_degree, y_degree-1)
                if x_degree + y_degree-1 < n-1:
                    boundary_matrix[basis_id, local_id] += -y_min-y_max
            if y_degree < n-1:
                local_id = self.binomial_to_local_id(x_i, x_degree, y_degree)
                if x_degree + y_degree < n-1:
                    boundary_matrix[basis_id, local_id] += y_min*y_max
"""
