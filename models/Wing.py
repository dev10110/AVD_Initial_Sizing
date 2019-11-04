from .Component import Component

class Wing(Component):
	"""Class contains wing components
	"""
	def __init__(self):

		self._loc_x = None
		self._loc_z = None
		self._sweep = None
		self._taperRatio = None
		self._area = None
		self._area_controlsurface = None
		self._aspectRatio = None
		self._dihedral = None
		self._thicknessToChord_root = None
	
	# ======================= getter /setter method ======================= 
	@property
	def loc_x(self):
		return self._loc_x

	@loc_x.setter
	def loc_x(self,a):
		self._loc_x = a


	@property
	def loc_z(self):
		return self._loc_y
	
	@loc_z.setter
	def loc_z(self,a):
		self._loc_z = a


	@property
	def sweep(self):
		return self._sweep

	@sweep.setter
	def sweep(self,a):
		self._sweep = a


	@property
	def taperRatio(self):
		return self._taperRatio

	@taperRatio.setter
	def taperRatio(self,a):
		self._taperRatio = a


	@property
	def area(self):
		return self._area
	
	@area.setter
	def area(self,a):
		self._area = a
	

	@property
	def area_controlsurface(self):
		return self._area_controlsurface
	
	@area_controlsurface
	def area_controlsurface(self,a):
		self._area_controlsurface = a
	

	@property	
	def aspectRatio(self):
		return self._aspectRatio

	@aspectRatio
	def aspectRatio(self,a):
		self._aspectRatio = a
	

	@property
	def dihedral(self):
		return self.dihedral
	
	@dihedral
	def dihedral(self,a):
		self._dihedral






	






