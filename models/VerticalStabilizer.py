class VerticalStabilizer:
	def __init__(self, x_percentage, sweep, area, aspectRatio):
		self.x_percentage = x_percentage
		self.sweep = sweep
		self.area = area
		self.aspectRatio = aspectRatio

	def __repr__(self):
		out = f'\n Object: {self.__class__.__name__}'

		for k in self.__dict__:
			out += f'\n{k} \t {self.__dict__[k]}'

		return out
