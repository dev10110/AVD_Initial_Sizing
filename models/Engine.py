class Engine:
	def __init__(self, root, bypassRatio, designThrust, diameter, number):
		# Vector
		self.root = root
		self.bypassRatio = bypassRatio
		self.designThrust = designThrust
		self.diameter = diameter
		self.number = number

	def __repr__(self):
		out = f'\n Object: {self.__class__.__name__}'

		for k in self.__dict__:
			out += f'\n{k} \t {self.__dict__[k]}'

		return out
