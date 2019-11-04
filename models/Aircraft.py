from .Component import Component

class Aircraft(Component):
	def __init__(self, fuselage, wing, engines, horizontalStabilizer, verticalStabilizer):
		self.fuselage = fuselage
		self.wing = wing
		self.engines = engines
		self.horizontalStabilizer = horizontalStabilizer
		self.verticalStabilizer = verticalStabilizer
