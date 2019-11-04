class HorizontalStabilizer:
	def __init__(self, x_percentage, sweep, taperRatio, fuselageDiameter, area, aspectRatio, areaRatio):
		self.x_percentage = x_percentage
		self.sweep = sweep
		self.taperRatio = taperRatio
		self.fuselageDiameter = fuselageDiameter
		self.area = area
		self.aspectRatio = aspectRatio
		self.areaRatio = areaRatio

	def __repr__(self):
		out = f'\n Object: {self.__class__.__name__}'

		for k in self.__dict__:
			out += f'\n{k} \t {self.__dict__[k]}'

		return out
