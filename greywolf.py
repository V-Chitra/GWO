
import numpy as np
import random
	

class solution:
	def __init__(self):
		self.best = 0
		self.bestIndividual=[]
		self.convergence = []



def binarize(x_d,ad ):
  """ Implements eqn 11, 12, 13"""
  cstep = 1/(1+np.exp(-10*(ad-0.5)))
  assert -0.01<cstep<1.01
  bstep = 1 if cstep>random.random() else 0
  return 1 if x_d+bstep >=1 else 0



def GWO_binary(objf,lb,ub,dim,SearchAgents_no=5,Max_iter=1000):
    
    # initialize alpha, beta, and delta_pos
    Alpha_pos=np.zeros(dim)
    Alpha_score=float("inf")
    
    Beta_pos=np.zeros(dim)
    Beta_score=float("inf")
    
    Delta_pos=np.zeros(dim)
    Delta_score=float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    #Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (np.random.uniform(lb[i],ub[i], SearchAgents_no) * (ub[i] - lb[i]) + lb[i] >= random.random())
    
    Convergence_curve=np.zeros(Max_iter)
    s=solution()

     # Loop counter
    print("GWO is optimizing  \""+objf.__name__+"\"")    
    
    # Main loop
    for l in range(0,Max_iter):
        for i in range(0,SearchAgents_no):
            
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i,j]=np.clip(Positions[i,j], lb[j], ub[j])


            # Calculate objective function for each search agent
            fitness=objf(Positions[i,:])
            #print(f"Iter: {l} Current fitness of wolf is {fitness}")
            # Update Alpha, Beta, and Delta
            if fitness<Alpha_score :
                Delta_score=Beta_score  # Update delte
                Delta_pos=Beta_pos.copy()
                Beta_score=Alpha_score  # Update beta
                Beta_pos=Alpha_pos.copy()
                Alpha_score=fitness; # Update alpha
                Alpha_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness<Beta_score ):
                Delta_score=Beta_score  # Update delte
                Delta_pos=Beta_pos.copy()
                Beta_score=fitness  # Update beta
                Beta_pos=Positions[i,:].copy()
            
            
            if (fitness>Alpha_score and fitness>Beta_score and fitness<Delta_score):                 
                Delta_score=fitness # Update delta
                Delta_pos=Positions[i,:].copy()
            
        
        
        
        a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0,SearchAgents_no):
            for j in range (0,dim):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*Alpha_pos[j]-Positions[i,j]); # Equation (3.5)-part 1
                X1=Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*Beta_pos[j]-Positions[i,j]); # Equation (3.5)-part 2
                X2=Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*Delta_pos[j]-Positions[i,j]); # Equation (3.5)-part 3
                X3=Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                X1 = binarize(X1, A1*D_alpha)
                X2 = binarize(X2, A2*D_alpha)
                X3 = binarize(X3, A3*D_alpha)
                

                Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
                exp = 1/(1+np.exp(-10*(Positions[i,j]-0.5)))
                #binarize
                Positions[i,j] = int(exp
                   >=random.random())


        Convergence_curve[l]=Alpha_score

        if (l):
               print(f"At iteration {l} the best fitness is {Alpha_score:.4f} for wolf: {Alpha_pos}");
    
    s.convergence=Convergence_curve
    s.optimizer="GWO"
    s.objfname=objf.__name__
    s.bestIndividual = Alpha_pos

    return s

