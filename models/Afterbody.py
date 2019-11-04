from .Component import Component

class Afterbody(Component):
	def __init__(self, diameter, afterbodyLDRatio):
		self.diameter = diameter
		self.afterbodyLDratio = afterbodyLDRatio
