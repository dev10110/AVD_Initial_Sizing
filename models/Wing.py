class Wing:

	def __init__(self, x_percentage, sweep, taperRatio, area, aspectRatio, dihedral):
		self.x_percentage = x_percentage
		# Must be of type engine that we defined!
		self.sweep = sweep
		self.taperRatio = taperRatio
		self.area = area
		self.aspectRatio = aspectRatio
		self.dihedral = dihedral

	def __repr__(self):
		out = f'\n Object: {self.__class__.__name__}'

		for k in self.__dict__:
			out += f'\n{k} \t {self.__dict__[k]}'

		return out
