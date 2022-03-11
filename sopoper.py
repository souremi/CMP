#!/usr/bin/env python3
"""
module sopoper

defines:
   class op1d: DOF-operator 

   subroutine txt2oper: build a sum-of-products operator
   

"""



import numbers
from numpy import *

class op1d:

    def __init__(self,N,op=1):
        """
        Create a 1D operator, requires grid
        dimension and operator. operator can be:

        Number: unit operator times number
        Vector: diagonal operator with vector on diagonal
        Matrix: full matrix operator

        Vector and matrix must be of type numy.ndarray

        Only real operators implemented, feel free to extend.

        """

        
        lnumber = isinstance(op, numbers.Number)

        if not lnumber and type(op) != ndarray:
            message = "class op1d: expecting number or numpy array"
            raise ValueError(message)
            
        self.N = N

        if lnumber:
            # unit operator times number
            self.hterm = diag(ones(self.N, dtype=double)*op)
        
            
        elif len(op.shape) == 1:
            # general diagonal operator
            if op.shape[0] != self.N:
                raise ValueError("Shape mismatch")
            self.hterm = diag(op)
            
        elif len(op.shape) == 2:
            # matrix operator
            if op.shape[0] != self.N or op.shape[1] != self.N :
                raise ValueError("Shape mismatch: op:"
                                 + str(op.shape) + "self: "
                                 + str((self.N,self.N)))

            self.hterm = op.copy()
            
        else:
            # unknown type
            message = "class op1d: unknown operator type."
            raise ValueError(message)
            
        
    def __mul__(self, other):

        # implement multiplication between two 1D operators
        # and multiplication with a number

        # require other be type op1d or number
        # feel free to extend
        
        if type(other) != type(self) and not isinstance(other, numbers.Number):
            return NotImplemented
    
        if not isinstance(other, numbers.Number):
            if self.N != other.N:
                message = "operators most be of same dimension: " + str(self.N) +"," + str(other.N)
                raise ValueError(message)
    
    
        # other is a number 
        if isinstance(other, numbers.Number):
            op = op1d(self.N, self.hterm)
            op.hterm *= other
            return op
                
                
        op = op1d(self.N, self.hterm @ other.hterm)
        return op

                
    def __rmul__(self, other):

        # mutliplication from right.
        # require other be a number.
        # feel free to extend
        
        if not isinstance(other, numbers.Number):
            return NotImplemented

        # other is a number
        op = op1d(self.N, self.hterm*other) 
        return op           
        
        
    def __add__(self, other):
        
        # add two op1d instances.
        # feel free to extend
        
        if type(other) != type(self):
            return NotImplemented
        
        return op1d(self.N, self.hterm + other.hterm)
    

    def operate(self, psi):
        """
        return H|psi> for H type(op1d), psi numpy 1D array
        with matching dimension. 
        """

        return self.hterm @ psi
        
        

    def expect(self, psi):
        """
        return <psi|H|psi> for H type(op1d), psi numpy 1D array
        with matching dimension. 
        assumes H hermitian,  <psi|H|psi> real
        """
        
        return vdot(psi,self.hterm @ psi).real
        
    


        
def txt2oper(txt, dvrlist, params={}):
    """
    Translate a text operator tableau with symbolic
    expressions into a sum-of-products operator
    using instances of class op1d.
    
    Input:
    
    txt     : string containing operator definition
    dvrlist : list of DVR objects
    params  : dictionary containing variable names
    and values for evaluation of the
    expressions for coefficients
    
    
    Format of txt:
    
    txt is organised as a table. 
    Each line of txt contains one product
    of operators. The first column of the
    table contains a numerical value, the
    coefficient, for instance -1/2m etc.
    The coefficient can be an arythmetic expression
    
    Each further column contains a symbolic
    expression for a 1D operator where the
    associated physical coordinate is identified
    by the dvr in the dvr list:
    
    Second column of txt is associated with
    first dvr from dvrlist etc.
    
    Columns in txt are separated by "|"
    Within each cell of the table
    multiplication with "*" may be used,
    but no other symbols like "+","-","/" etc.
    are allowed.
    (for "+","-" multiply out and use new lines)
    
    Within the coefficient column also
    any other python arythmetic may be used.
    
    Example: 2D coupled harmonic oscillator
    H = -1/2m d^2 / dx^2  -1/2m d^2 / dy^2 + 1/2 m omega^2 (x^2 + y^2) + alpha*x*y 
    
    # coeff        | x         | y
    #--------------------------------
    -0.5/m         | dq^2      | 1
    -0.5/m         | 1         | dq^2
    0.5*m*omega**2 | q^2       | 1  
    0.5*m*omega**2 | 1         | q^2  
    alpha          | q         | q
    
    Empty lines and everything after
    "#" in a line is  ignored.
    
    The parameters "m", "omega", and "alpha"
    must be defined as strings in the params dictionary,
    for instance:
        
    params={} # empty dictionary
    params["m"] = 1
    params["omega"] = 1
    params["alpha"] = 0.2

    Built-in symbols:
        "1"     : unit operator (identity)
        "q"     : coordinate operator
        "dq"    : first derivative
        "dq^2"  : second derivative
        "q^-1"  : 1/q
        "q^-2"  : 1/q**2
        "q^2"   : q**2
        "qs1"   : sqrt(1-q**2)
        "qs1^2" : 1-q**2
        
        (feel free to extend!
         also exponent parsing might
         be a useful thing)
        
    """


    lines = txt.split("\n")
    
    operator = []

    for line in lines:
        # remove leading and trailing whitespaces
        line = line.strip()
        
        # ignore all after a #
        if "#" in line:
            i = line.index("#")
            line = line[:i]
            
        # skip empty lines
        if line == "": continue 
        
        # coefficient gets special treetment
        terms = line.split("|")
        coeff = terms.pop(0)
        
        # evaluate coefficients with parameters
        # to a number
        coeff = eval(coeff,params)
        
        hproduct = [coeff,]
        
        if len(terms) != len(dvrlist):
            message = "Number of terms must match number of DVRs!"
            raise ValueError(message)
      
        
        for f, term in enumerate(terms):
            term = term.strip()
            
            if term == "":
                message= "No epmty terms allowed!"
                raise ValueError(message)
                
            factors = term.split("*")
            
            result = op1d(dvrlist[f].N) # unit operator
            
            # build operators
            for factor in factors:
                
                if factor == "1":
                    op = op1d(dvrlist[f].N, 1)
                    result = result*op
                
                elif factor == "q":
                    q = dvrlist[f].grid
                    op = op1d(dvrlist[f].N, q)
                    result = result*op
                
                elif factor == "dq":
                    dq = dvrlist[f].dif1.copy()
                    op = op1d(dvrlist[f].N, dq)
                    result = result*op
                
                elif factor == "dq^2":
                    dq2 = dvrlist[f].dif2.copy()
                    op = op1d(dvrlist[f].N, dq2)
                    result = result*op
            
                elif factor == "q^-1":
                    q1 = dvrlist[f].grid**-1
                    op = op1d(dvrlist[f].N, q1)
                    result = result*op
                        
                elif factor == "q^-2":
                    q2 = dvrlist[f].grid**-2
                    op = op1d(dvrlist[f].N, q2)
                    result = result*op
                    
                elif factor == "q^2":
                    q2 = dvrlist[f].grid**2
                    op = op1d(dvrlist[f].N, q2)
                    result = result*op

                elif factor == "qs1":
                    qs = sqrt(1.0 - dvrlist[f].grid**2) 
                    op = op1d(dvrlist[f].N, qs)
                    result = result*op
                    
                elif factor == "qs1^2":
                    qs2 = 1.0 - dvrlist[f].grid**2 
                    op = op1d(dvrlist[f].N, qs2)
                    result = result*op
                    
                else:
                    message = "operator not implemented: "+factor
                    raise ValueError(message)
                    
            hproduct.append(result)
        operator.append(hproduct)
    return operator
    
