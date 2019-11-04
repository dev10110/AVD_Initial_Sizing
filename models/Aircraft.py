from .Component import Component

class Aircraft(Component):
	def __init__(self, fuselage, wing, engines, tail):

		self.fuselage = fuselage
		self.wing = wing
		self.engines = engines
		self.tail = tail


		self.TW = None
		self.MTOW = None

	@property
	def TW(self):
		self._TW = self.engines._thrust/self._MTOW
		return self._TW

	@TW.setter
	def TW(self, TW):
		raise AssertionError('Cant Set TW directly')

	@property
	def MTOW(self):
		#self._MTOW =
		raise NotImplementedError()
