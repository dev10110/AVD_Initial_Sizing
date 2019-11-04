from .Component import Component

class VerticalStabilizer(Component):
	def __init__(self, x_percentage, sweep, area, aspectRatio):
		self.x_percentage = x_percentage
		self.sweep = sweep
		self.area = area
		self.aspectRatio = aspectRatio
