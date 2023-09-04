import numpy as np
from itertools import product
from scipy.stats import unitary_group
import random

X=np.array([[0,1],[1,0]],dtype=complex)
Y=np.array([[0,-1j],[1j,0]],dtype=complex)
Z=np.array([[1,0],[0,-1]],dtype=complex)

class qstate:
    def __init__(self,n=1,random=False,arr=[]):
        
        self.nqubits=n
        
        if(random==True):
            self.arr=np.random.normal(size=2**n)+1j*np.random.normal(size=2**n)
            self.arr=self.arr/np.linalg.norm(self.arr)
            
        elif(random==False and len(arr)==0):
            self.arr=np.zeros(2**n,dtype=complex)
            self.arr[0]=1
            
        elif(len(arr)!=0):
            self.arr=np.array(arr,dtype=complex)/np.linalg.norm(arr)
            self.nqubits=int(np.log2(len(arr)))
            
        self.norm=np.linalg.norm(self.arr)
    
    
    def apply_1gate(self,gate,i=0,inplace=True,p1=None):

        #apply gate on qubit i (0 indexed)
        
        if(inplace==False):
            arr=self.arr.copy()
        
        gate=np.array(gate,dtype=complex)

        # if noisy, apply a gate randomly chosen from X,Y,Z with the same probability
        if p1 is not None:
            if(np.random.rand()<p1):
                gate=gate.dot(random.choice([X,Y,Z]))

        
        L=self.nqubits

        # reshape into (2,2,2,..) array so that contractions are easier
        self.arr=np.reshape(self.arr,tuple([2]*L))
        
        #contract gate with state
        self.arr=np.einsum(gate,[L,i],self.arr,range(L),[j for j in range(i)]+[L]+[j for j in range(i+1,L)])
        
        # revert to original shape
        self.arr=np.reshape(self.arr,2**self.nqubits)
        
        if(inplace==True):
            self.norm=np.linalg.norm(self.arr)
            return None
        else:
            new=self.arr
            self.arr=arr
            return new
    
    def apply_2gate(self,gate,i=0,j=1,inplace=True,p2=None):
        
        
        if(self.nqubits<2):
            raise ValueError("Number of qubits smaller than 2!")
            
        if(inplace==False):
            arr=self.arr.copy()

        # if input indices not ordered correctly, reorder
        if(i>j):
            tmp=i
            i=j
            j=tmp

        # if noisy, apply a gate randomly chosen from two qubit Pauli Products with the same probability
        if p2 is not None:
            if(np.random.rand()<p2):
                error_1=random.choice([np.eye(2),X,Y,Z])

                if(np.linalg.norm(error_1-np.eye(2))==0): # if first qubit error is I, exclude I as a possible choice in 2nd qubit error
                    error_2 = random.choice([X,Y,Z])
                else:
                    error_2 = random.choice([np.eye(2),X,Y,Z])

                gate=gate.dot(np.kron(error_1,error_2))

        
        L=self.nqubits 

        # reshape state to (2,2,2...)
        self.arr=np.reshape(self.arr,tuple([2]*L))

        # reshape two qubit gate to (2,2,2,2) (C ordering)
        gate=np.array(gate,dtype=complex)
        gate=np.reshape(gate,tuple([2]*4))

        # contract gate with state
        self.arr=np.einsum(gate,[L,L+1,i,j],self.arr,range(L),[k for k in range(i)]+[L]+[k for k in range(i+1,j)]+[L+1]+[k for k in range(j+1,L)])

        # restore shape
        self.arr=np.reshape(self.arr,2**self.nqubits)

        if(inplace==True):
            self.norm=np.linalg.norm(self.arr)
            return None
        else:
            new=self.arr
            self.arr=arr
            return new

        
        
        
    def measure(self,basis_list=['Z'],only_subsystem=False,inplace=True):
        

        if(basis_list==['I']*self.nqubits): #if no qubits are being measured
            return None
        
        
        # if non-standard, assume all qubits are being measured in Z basis
        if(len(basis_list)!=self.nqubits):
            basis_list=['Z']*self.nqubits
        
        
        # if basis element 'I', qubit is NOT measured
        meas_qubits=[]

        if(inplace==False):
            pre_arr=self.arr[:]
        
        for i in range(self.nqubits):
            
            if(basis_list[i]=='X'):
                self.arr.apply_1gate(X,i)
                meas_qubits.append(i)
            elif(basis_list[i]=='Y'):
                self.arr.apply_1gate(Y,i)
                meas_qubits.append(i+1)
                
            elif(basis_list[i]=='Z'):
                meas_qubits.append(i)
                
        all_qubits=[i for i in range(self.nqubits)]
        
        
        n_meas=len(meas_qubits)

        traced_qubits=np.setdiff1d(np.arange(self.nqubits),meas_qubits)

        
        shape_arr=np.repeat(2,self.nqubits)
        psi=np.reshape(self.arr,shape_arr)

        if(n_meas!=self.nqubits):
            prob_arr=np.sum(np.abs(psi)**2,axis=tuple(traced_qubits)) #tracing out qubits that won't be measured
        
        else:
            prob_arr=np.abs(psi)**2

        # reshape to sample integer
        prob_arr=np.reshape(prob_arr,2**n_meas)

        meas_int=np.random.choice(range(2**n_meas),p=prob_arr) #choosing the computational basis state randomly after measurement based on 
        meas_str = format(meas_int,'0'+str(n_meas)+'b')

        
        # initialising collapsed state on the unmeasured system
        if(only_subsystem):
            psi_b=np.zeros(2**(self.nqubits-n_meas),dtype="complex")

        else:
            psi_b=np.zeros(2**self.nqubits,dtype="complex")
            

        # find the full collapsed state including the unmeasured qubits
        for i in range(2**(self.nqubits-n_meas)):
            
            # find the full binary string inserting the measured string
            bin_i=format(i,'0'+str(self.nqubits-n_meas)+'b')
            bin_full=['0']*self.nqubits
            
            for j,m in enumerate(meas_qubits):
                bin_full[m]=meas_str[j]
                
            for j,m in enumerate(traced_qubits):
                bin_full[m]=str(bin_i[j])

            new_i=int("".join(bin_full),2)

            if(only_subsystem):
                psi_b[i]=self.arr[new_i]
                
            else:
                psi_b[new_i]=self.arr[new_i]
                
            
        
        # update number of qubits and state array
        if(only_subsystem):
            self.nqubits=self.nqubits-n_meas
                             
        self.arr=psi_b/np.linalg.norm(psi_b) # normalize                    

        if(inplace==False):
            new=self.arr
            self.arr=pre_arr
                    
        return meas_qubits,meas_str
    
    
    def compute_ent_ee(self,subsystem_size = None,svd_thresh=1e-8) :
        
        if(subsystem_size==None):
            subsystem_size=self.nqubits//2
        
        # reshape psi so singular values can be computed
        psi=np.reshape(self.arr,(2**(subsystem_size),2**(self.nqubits-subsystem_size)))
        
        s=np.linalg.svd(psi,compute_uv=False)

        ent_entropy = 0
        for j in range(len(s)):
            if(s[j]>svd_thresh): # ignoring very small values
                ent_entropy=ent_entropy-np.log(s[j]**2)*(s[j]**2)
    
    
        return ent_entropy
    
    

    def random_brickwork_evolve(self,nsteps=1,pbc=True,p_meas=None):
        #periodic boundary conditions by default
        
        if(pbc==True and self.nqubits%2!=0):
            return "For periodic boundary conditions, need an even number of qubits"
        
        
        for step in range(nsteps):
        
            # layer 1 for even numbered qubits
            
            for m in range(0,self.nqubits,2):
                
                twogate=unitary_group.rvs(4)
                self.apply_2gate(gate=twogate,i=m,j=m+1)
                
            # layer 2 for odd numbered qubits
            
            for m in range(1,self.nqubits-1,2):
                
                twogate=unitary_group.rvs(4)
                self.apply_2gate(gate=twogate,i=m,j=m+1)            
        
            # if pbc add a gate between first and last in layer 2
            if(pbc==True and self.nqubits>2):
                twogate=unitary_group.rvs(4)
                self.apply_2gate(gate=twogate,i=0,j=self.nqubits-1)
                
                
            
            # measure the state after each layer with a probability p_meas
            if(p_meas != None):
                meas_basis=[]
                for i in range(self.nqubits):
                    if(np.random.rand()<p_meas):
                        meas_basis.append('Z')
                        
                    else:
                        meas_basis.append('I')
                        
                self.measure(meas_basis)

        return None

    def overlap(self,psi):

        return np.conjugate(np.transpose(psi.arr)).dot(self.arr)
