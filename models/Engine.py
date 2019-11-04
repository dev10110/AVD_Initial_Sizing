from .Component import Component
class Engine(Component):
	def __init__(self, root, bypassRatio, designThrust, diameter, number):
		# Vector
		self.root = root
		self.bypassRatio = bypassRatio
		self.designThrust = designThrust
		self.diameter = diameter
		self.number = number
