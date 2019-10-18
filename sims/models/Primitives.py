import numpy as np    

class Vector:

  def __init__(self, x, y, z):
    self.data = np.array([x,y,z])