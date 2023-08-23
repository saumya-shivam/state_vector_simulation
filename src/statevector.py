import numpy as np
from itertools import product
from scipy.stats import unitary_group

X=np.array([[0,1],[1,0]],dtype=complex)
Y=np.array([[0,-1j],[1j,0]],dtype=complex)
Z=np.array([[1,0],[0,-1]],dtype=complex)

class qstate:
    def __init__(self,n=1,random=False,arr=[]):
        
        self.nqubits=n
        
        if(random==True):
            self.arr=np.random.rand(2**n)+1j*np.random.rand(2**n)
            self.arr=self.arr/np.linalg.norm(self.arr)
            
        elif(random==False and len(arr)==0):
            self.arr=np.zeros(2**n,dtype=complex)
            self.arr[0]=1
            
        elif(len(arr)!=0):
            self.arr=np.array(arr,dtype=complex)/np.linalg.norm(arr)
            self.nqubits=int(np.log2(len(arr)))
            
        self.norm=np.linalg.norm(self.arr)
    
    
    def apply_1gate(self,gate,i=0,inplace=True):

        #apply gate on qubit i (0 indexed)
        
        Ntot=2**self.nqubits # total dimesnions of the Hilbert space

        arr=self.arr.copy()
        gate=np.array(gate,dtype=complex)
        
        # say if m ranges from 0 to Ntot-1, mp ranges from 0 to Ntot/2 -1 since ith position bit string is fixed

        L=self.nqubits
        
        alpha,beta=0,1 # two states which can be generalized for qudits later
        
        for mp in range(Ntot//2):

            # suppose mp = m1'+m2', where m1'=b1x2**(L-2)+b2x2**(L-3)+... +b_{i}x2**(L-i-1), m2'=b_{i+2}2**(L-i-2)+...+b_{L-1}2**(1)+b_{L}q**(0)
            # then by definition m1' = (2**(L-i-1))*mp//2**(L-i-1)
            # and m2'=mp-m1'
            m1p=(2**(L-i-1))*(mp//(2**(L-i-1)))
            m2p=mp-m1p

            # now the two states which will be modified are ones which have alpha and beta at ith position

            alpha_ind=2*m1p+alpha*(2**(L-i-1))+m2p

            beta_ind=2*m1p+beta*(2**(L-i-1))+m2p

            # modifying the state at the corresponding indices

            arr[alpha_ind]=self.arr[alpha_ind]*gate[0,0]+gate[0,1]*self.arr[beta_ind]
            arr[beta_ind]=self.arr[beta_ind]*gate[1,1]+gate[1,0]*self.arr[alpha_ind]

        if(inplace==True):
            self.arr=arr
            self.norm=np.linalg.norm(self.arr)
            return None
        else:
            return arr
    
    def apply_2gate(self,gate,i=0,j=1,inplace=True):
        
        if(self.nqubits<2):
            return print("Number of qubits smaller than 2!")
                
        # pick four levels at the two sites, for qubits by default 0 and 1
        a1,b1= 0,1
        a2,b2= 0,1


        Ntot=2**self.nqubits # total dimesnions of the Hilbert space

        arr=self.arr.copy()
        gate=np.array(gate,dtype=complex)

        # say if m ranges from 0 to Ntot-1, mp ranges from 0 to Ntot/q**2 -1 since ith and jth position dit string is fixed
        # important that i<j!

        if(i>j):
            tmp=i
            i=j
            j=tmp

        q=2 # for qubits
        L=self.nqubits 
        
        for mp in range(Ntot//(q**2)):

            # suppose mp = m1'+m2'+m3', 
            # where m1'=b1q**(L-3)+b2q**(L-4)+... +b_{i-1}q**(L-i-1), m2'=b_{i+1}q**(L-i-2)+ ...+b_{j-1}q**(L-j)
            # ,m3'=b_{j+1}q**(L-j-1).....+b_{L-1}q**(1)+b_{L}q**(0)
            # then by definition m1' = (q**(L-i-1))*mp//q**(L-i-1)
            # and m2'=(q**(L-j))*mp//q**(L-j)-m1'
            # and m3'= m'-m1'-m2'

            m1p=(q**(L-i-2))*(mp//(q**(L-i-2)))
            m2p=(q**(L-j-1))*(mp//(q**(L-j-1)))-m1p
            m3p=mp-m1p-m2p

            # now the four states which will be modified are ones which have alpha and beta at ith position

            a1a2_ind=(q**2)*m1p+a1*(q**(L-i-1))+q*m2p+a2*(q**(L-j-1))+m3p
            a1b2_ind=(q**2)*m1p+a1*(q**(L-i-1))+q*m2p+b2*(q**(L-j-1))+m3p            
            b1a2_ind=(q**2)*m1p+b1*(q**(L-i-1))+q*m2p+a2*(q**(L-j-1))+m3p
            b1b2_ind=(q**2)*m1p+b1*(q**(L-i-1))+q*m2p+b2*(q**(L-j-1))+m3p

            
            psi_short=np.array([self.arr[a1a2_ind],self.arr[a1b2_ind],self.arr[b1a2_ind],self.arr[b1b2_ind]])
            
            arr[a1a2_ind]=gate[0,:].dot(psi_short)
            arr[a1b2_ind]=gate[1,:].dot(psi_short)
            arr[b1a2_ind]=gate[2,:].dot(psi_short)
            arr[b1b2_ind]=gate[3,:].dot(psi_short)        

        if(inplace==True):
            self.arr=arr
            self.norm=np.linalg.norm(self.arr)
            return None
        else:
            return arr

        
        
        
    def measure(self,basis_list=['Z'],only_subsystem=False):
        

        if(basis_list==['I']*self.nqubits): #if no qubits are being measured
            return None
        
        
        # if non-standard, assume all qubits are being measured in Z basis
        if(len(basis_list)!=self.nqubits):
            basis_list=['Z']*self.nqubits
        
        
        # if basis element 'I', qubit is NOT measured
        meas_qubits=[]

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
                bin_full[m]=meas_str[j]#str(meas_ind[j])
                
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
