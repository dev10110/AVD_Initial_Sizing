class Nose():
	def __init__(self, diameter, noseLDRatio):

		self.diameter = diameter
		self.noseLDRatio = noseLDRatio

	def __repr__(self):
		out = f'\n Object: {self.__class__.__name__}'

		for k in self.__dict__:
			out += f'\n{k} \t {self.__dict__[k]}'

		return out
