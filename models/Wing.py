class Wing:

	def __init__(self, x_percentage, engines, sweep, taperRatio, area, aspectRatio, dihedral):
		self.x_percentage = x_percentage
		# Must be of type engine that we defined!
		self.engines = engines
		self.sweep = sweep
		self.taperRatio = taperRatio
		self.area = area
		self.aspectRatio = aspectRatio
		self.dihedral = dihedral
