class Aircraft:
	def __init__(self, fuselage, wing, engines, horizontalStabilizer, verticalStabilizer):
		self.fuselage = fuselage
		self.wing = wing
		self.engines = engines
		self.horizontalStabilizer = horizontalStabilizer
		self.verticalStabilizer = verticalStabilizer

	def __repr__(self):
		out = f'\n Object: {self.__class__.__name__}'

		for k in self.__dict__:
			out += f'\n{k} \t {self.__dict__[k]}'

		return out
