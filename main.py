import random
import numpy
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from greywolf import GWO_binary, solution


from sklearn.linear_model import LogisticRegression



class DATA:
  df = pd.read_csv('diabetes_data.csv', names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])

  #It is a helper function
  def _lrscore(x_vec, y):
      clf = LogisticRegression(random_state=0).fit(x_vec, y)
      return(clf.score(x_vec, y))

  #Returns error for the LR fit given set of features 
  # --> Call this function from GWO
  def errorReturn(feature_selected):
    feature_selected = feature_selected.astype(int)

    #print(feature_selected, '....')
    f = []
    [
        f.append(DATA.df.columns[i]) for i in range(len(feature_selected))
        if feature_selected[i] == 1
    ]
    if not f: return 1

    x_vec = np.array(DATA.df[f])
    y = np.array(DATA.df.loc[:, 'class'])

    return(1-DATA._lrscore(x_vec,y)) #minimize error


def convergence_plot(data_from_multiple_runs, optimizer_name):
      for d in data_from_multiple_runs:
        plt.plot(d)
      plt.title(f"{optimizer_name} plot")
      plt.show()
      return
		

def featureSelectionGWO(df, SearchAgents_no=5, Max_iter=30):
  dim = len(df.columns)-1

  sol = GWO_binary(
    DATA.errorReturn, 
  lb=0,
  ub=1,
  dim=dim,
  SearchAgents_no=SearchAgents_no,
   Max_iter=Max_iter)

  #convergence_plot(sol.convergence, 'gwo_binary')

  #feature_selected = [1, 1, 1, 1, 1, 1, 1, 1]
  return sol.bestIndividual



#Optimum values for feature selected are
# [1,0,0,0,0,1,1,0] and [0, 1, 0, 0, 1, 1, 0, 1]
if __name__ == '__main__':
    f = featureSelectionGWO(DATA.df, 5)
    print(f"{f}: score {DATA.errorReturn(np.array(f))}")
    print(f"{f}: score {DATA.errorReturn(np.array(f))}")


