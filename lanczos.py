
from numpy import *
from scipy.linalg import eigh_tridiagonal

import warnings


class eigh_lanczos:
    """
    class eigh_lanczos(H,psi0,nwanted,maxorder,tol=1e-08)

    Input:
        H: Linear operator, must have 'matvec' member function to operate on psi0
        
        psi0: start vector for Lanczos recursion

        nwanted: Number of eigenvalues and eigenvectors to be computed

        maxorder: maximum number of Lanczos recursion

        tol: consider an eigenenergy En converged if En changes less then
             tol when one more more krylov vector is used.


    """
    
    def __init__(self,H,psi0,nwanted,maxorder,tol=1e-08):
        

        self.maxorder =  maxorder
        self.nwanted = nwanted
        self.H = H
        
        self.M = product(psi0.shape)


        if self.maxorder > self.M:
            message="eigh_lanczos: maxorder must be <= number of elements in psi."
            raise ValueError(message)
        if self.nwanted > self.maxorder:
            message="eigh_lanczos: nwanted must be <= maxorder."
            raise ValueError(message)

        
        psi = psi0.reshape((self.M,))

        # krylov vectors and Hamilton matrix elements
        self.krylov = zeros((self.maxorder,self.M), dtype=complex)
        self.alpha = zeros(self.maxorder, dtype=double)
        self.beta = zeros(self.maxorder, dtype=double)
        
    
        norm = vdot(psi,psi)
        phi = psi/sqrt(norm)

    
        self.krylov[0,:] = phi

    
        # start the recursion
        self.r = H.matvec(self.krylov[0,:])
        self.alpha[0] = vdot(self.r,self.krylov[0,:]).real  # = <psi|H|psi>
        self.r = self.r - self.alpha[0]*self.krylov[0,:]   
        self.nkrylov = 1
        
        stoprecursion = False
        # recursion
        for n in range(1,self.maxorder):
            
            self.beta[n-1] = sqrt(vdot(self.r,self.r).real)
            
            # we are converged, no more eigenvectors is psi0
            if self.beta[n-1] > tol:
                
            
                phi = self.r/self.beta[n-1]  # = r/sqrt(<r|r>)
            
            
                # phi is the next krylov vector
                # and should be orthogonal
                # to all previous ones. but there
                # will be round-off errors
                # for numerical stability
                # we re-orthogonalize
                # to all previous vectors
                # using Schmidt-Othogonalization
                
                for i in range(n):
                    ctmp = vdot(self.krylov[i,:],phi)
                    phi = phi - ctmp*self.krylov[i,:]
                    
                    
                norm = vdot(phi,phi)
                self.krylov[n,:] = phi/sqrt(norm)
                
                self.r = self.H.matvec(phi)
                
                self.alpha[n] = vdot(self.r,phi).real  # = <phi|H|phi> 
                
                self.r = self.r - self.alpha[n]*phi  -  self.beta[n-1]* self.krylov[n-1,:]
                
                self.nkrylov += 1


                    
            else:
                stoprecursion = True
               


            # we need to get rid of spurious eigenvalues:
            # compare the two results and return only 
            # those that do not appear in the matrix 
            # with first row and column missing
            if n < self.nwanted-1 and not stoprecursion:
                
                # we need at least nwanted vectors
                # so skip error checking 
                continue
          
        
            else:
                # remove last row and column and first row and column to check convergence
                # this is usually cheap as the lanczos matrix is unually not large: N = 10^1 to 10^2
                # also check for matrix with one less krylov vector for convergenve
                # (could have neen stored from last iteration, actually)
                # we don't need eigenvectors at this point
                evals1     = eigh_tridiagonal(self.alpha[:n-1],self.beta[:n-2],  eigvals_only=True)
                evals      = eigh_tridiagonal(self.alpha[:n]  ,self.beta[:n-1],  eigvals_only=True)
                evals_rem1 = eigh_tridiagonal(self.alpha[1:n] ,self.beta[1:n-1], eigvals_only=True)
                                
                validevals = zeros(len(evals), dtype=double)
                index =  zeros(len(evals), dtype=int)
                
                nvalid = 0

                # loop over evals and copy only those 
                # that do not appear in evals_rem1

                for n1,En in enumerate(evals):
                    if not any(isclose(evals_rem1, En, rtol=1.e-8)):
                        # En is not in the evals_rem1 list
                        
                        validevals[nvalid] = En
                        index[nvalid] = n1
                        nvalid += 1
                        
                
                
                # we have a new krylov space with one vector more
                # and the Hamiltonian matrix as the alphas and betas
                # with one more row and column
                # now check eigenvalues until nwanted are stable
                # for this compare with values to the previous matrix: evals1
                
                if nvalid < self.nwanted:
                    
                    if n < self.maxorder-1:
                        continue # next n loop
                    else:
                        message ="""
                        Not converged: could only retrieve {} eigenvalues
                        of wanted {} with given start vector.""".format(nvalid, self.nwanted)
                        
                        warnings.warn(message)


                # check if we have self.nwanted stable eigenvalues

                self.nstable = 0
                self.stable = zeros(nvalid, dtype=int)
                # self.stable[:] = NaN # debug
                self.estable = zeros(nvalid, dtype=double)
                
                

                for i in range(nvalid):
                    n1 = index[i]
                   
                    if any(isclose(evals1, evals[n1], rtol=tol)):
                        self.estable[self.nstable] = evals[n1]
                        self.stable[self.nstable]= n1
                        self.nstable += 1
                                 
                
                if self.nstable >= self.nwanted or stoprecursion or n == self.maxorder -1 :
                    
                    evals, evecs = eigh_tridiagonal(self.alpha[:n],self.beta[:n-1], eigvals_only=False)
                                        
                    self.evals = zeros(self.nstable, dtype=double)
                    self.evecs = zeros((self.nstable,self.M), dtype=complex)
                    evecs_grid = evecs.T @ self.krylov[:n,:]
                    
                    for i in range(self.nstable):
                        n1 = self.stable[i]
                   
                        self.evals[i] = evals[n1]
                        self.evecs[i,:] = evecs_grid[n1,:]

                    if self.nwanted > self.nstable:
                        message = """Could only retrieve {} eigenvalues
                        of wanted {} with given start vector. Increase maxorder
                        or use different start vector?""".format(self.nstable, self.nwanted)
                        warnings.warn(message)
                    
                    return
  
    def report(self):

        message="""
        Number of state found:        {}
        Recursion order:              {}
        """.format(self.nstable,self.nkrylov)
        print(message)





def SILintegrator(H,psi0,dt,maxorder,tol=1e-08):
    """
    SILintegrator(H,psi0,dt,maxorder,tol=1e-08)

    Input:
        H: Linear operator, must have 'matvec' member function to operate on psi0
        
        psi0: start vector for Lanczos recursion

        dt: time step

        maxorder: maximum number of Lanczos recursions

        tol: continue recursions until estimated error is below tol.

    """


    M = product(psi0.shape)
    
    if maxorder > M:
        message="SILintegrator: maxorder must be <= number of elements in psi."
        raise ValueError(message)
            
    psi = psi0.reshape((M,))
        
    # krylov vectors and Hamilton matrix elements
    krylov = zeros((maxorder,M), dtype=complex)
    alpha = zeros(maxorder, dtype=double)
    beta = zeros(maxorder, dtype=double)
        
    
    norm = vdot(psi,psi)
    phi = psi/sqrt(norm)
    
    krylov[0,:] = phi

    
    # start the recursion
    r = H.matvec(krylov[0,:])
    alpha[0] = vdot(r,krylov[0,:]).real  # = <psi|H|psi>
    r = r - alpha[0]*krylov[0,:]   
    nkrylov = 1
    
    error = dt # beta_0 = 1
    

    # recursion
    for n in range(1,maxorder):
            
        beta[n-1] = sqrt(vdot(r,r).real)
        error *= dt*beta[n-1]/n

        if beta[n-1] > tol:
            phi = r/beta[n-1]  # = r/sqrt(<r|r>)
            
            # phi is the next krylov vector
            # and should be orthogonal
            # to all previous ones. But there
            # will be round-off errors
            # for numerical stability
            # we could re-orthogonalize
            # to all previous vectors
            # using Schmidt-Othogonalization
            
            #for i in range(n):
            #    ctmp = vdot(krylov[i,:],phi)
            #    phi = phi - ctmp*krylov[i,:]
                
                
            krylov[n,:] = phi
            
            r = H.matvec(phi)
            
            alpha[n] = vdot(r,phi).real  # = <phi|H|phi> 
            
            r = r - alpha[n]*phi  -  beta[n-1]* krylov[n-1,:]
            
            nkrylov += 1

        
        if error < tol or nkrylov >= maxorder or beta[n-1] < tol :
            # we are converged
            
            if error > tol:
                message = "Error > tol. Increase maxorder or decrease dt?"
                warnings.warn(message)

            
            evals, evecs = eigh_tridiagonal(alpha[:n],beta[:n-1], eigvals_only=False)

            # no sorting of invalid eigenvectors here
            # the recursion is not very deep, they should
            # not appear
            
            propagator = evecs[0,:]*exp(-1j*dt*evals) # Z_0k *exp(-i E_k dt)
            propagator = evecs @ propagator           # a_j

           
            # transform back
            psiplusdt = zeros(M, dtype=complex)
            for i in range(n):
                psiplusdt += propagator[i]*krylov[i,:] # sum_j a_j phi_j
            
    
            # stops loop  
            return psiplusdt
                
                























