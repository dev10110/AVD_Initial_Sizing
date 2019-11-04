# Defines parameters for the main body fuselage

class CentreFuselage:
	def __init__(self, diameter, length):
		self.diameter = diameter
		self.length = length

	def __repr__(self):
		out = f'\n Object: {self.__class__.__name__}'

		for k in self.__dict__:
			out += f'\n{k} \t {self.__dict__[k]}'

		return out
