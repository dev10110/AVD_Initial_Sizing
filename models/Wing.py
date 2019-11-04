from .Component import Component

class Wing(Component):

	def __init__(self, x_percentage=None, sweep=None, taperRatio=None, area=None, aspectRatio=None, dihedral=None):

		self.x_percentage = x_percentage
		# Must be of type engine that we defined!
		self.sweep = sweep
		self.taperRatio = taperRatio
		self.area = area
		self.aspectRatio = aspectRatio
		self.dihedral = dihedral
