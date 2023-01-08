import torch

_min_iters    = 3
_max_iters    = 1000
_history_size = 100

def fix_point(f, x0, nb_it_max = 1e2, tol=1e-10):
        x = x0
        nb_it = 0
        while torch.norm(x - f(x))>tol and nb_it<nb_it_max:
                #x_prev = x.clone()
                x = f(x)
                nb_it += 1
        
        return x           


def anderson(f, x0, m=5, lam=1e-4, nb_it_max=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d = x0.shape
    X = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0, f(x0)
    X[:,1], F[:,1] = F[:,0], f(F[:,0])
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    #res = []
    for k in range(2, nb_it_max):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        #res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        res = (F[:,k%m] - X[:,k%m]).norm().item()
        #if (res[-1] < tol):
        if (res < tol):
            break
    return X[:,k%m].view_as(x0)#, res

# Taken from https://github.com/vreshniak/ImplicitResNet/
def lbfgs( fun, x0, tol=None, max_iters=_max_iters, min_iters=_min_iters, history_size=_history_size, batch_error='max' ):
	iters = [0]
	error = [0]
	flag  = 0

	if batch_error=='max':
		batch_err = lambda z: z.amax()
	elif batch_error=='mean':
		batch_err = lambda z: z.mean()

	dtype  = x0.dtype
	device = x0.device

	if tol is None: tol = 10*torch.finfo(dtype).eps

	# check initial residual
	with torch.no_grad():
		error[0] = batch_err(fun(x0))
		if error[0]<tol: return x0.detach(), error[0].detach(), iters[0], flag

	# initial condition: make new (that's why clone) leaf (that's why detach) node which requires gradient
	x = x0.clone().detach().requires_grad_(True)

	nsolver = torch.optim.LBFGS([x], lr=1, max_iter=max_iters, max_eval=None, tolerance_grad=1.e-12, tolerance_change=1.e-12, history_size=history_size, line_search_fn='strong_wolfe')
	def closure():
		resid = fun(x)
		error[0] = batch_err(resid)
		residual = resid.mean()
		nsolver.zero_grad()
		# if error[0]>tol: residual.backward()
		# use .grad() instead of .backward() to avoid evaluation of gradients for leaf parameters which must be frozen inside nsolver
		if error[0]>tol or iters[0]<min_iters: x.grad, = torch.autograd.grad(residual, x, only_inputs=True, allow_unused=False)
		iters[0] += 1
		return residual
	nsolver.step(closure)

	if error[0]>tol: flag=1

	return x.detach(), error[0].detach(), iters[0], flag

def gradient_descent(f, grad, x0, lr=0.1, n_iter=100, tol=1e-10, momentum=0.9):
    x = x0
    diff = torch.zeros(x.shape).type_as(x)

    for _ in range(n_iter):            
        if torch.norm(f(x)) <= tol:
            break
        diff = momentum*diff -lr * 2 * torch.matmul(torch.transpose(grad(x), dim0=-1, dim1=-2), f(x).unsqueeze(-1)).squeeze()
        x = x + diff                    
    
    return x
