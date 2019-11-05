from .Component import Component

class Wing(Component):

	def __init__(self):

		self._loc_x = None
		self._loc_z = None
		self._sweep = None
		self._taperRatio = None
		self._area = None
		self._aspectRatio = None
		self._dihedral = None
