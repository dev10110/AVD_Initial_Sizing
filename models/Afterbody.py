class Afterbody:
	def __init__(self, diameter, afterbodyLDRatio):
		self.diameter = diameter
		self.afterbodyLDratio = afterbodyLDRatio

	def __repr__(self):
		out = f'\n Object: {self.__class__.__name__}'

		for k in self.__dict__:
			out += f'\n{k} \t {self.__dict__[k]}'

		return out
