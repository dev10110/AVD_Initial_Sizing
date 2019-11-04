from .Component import Component

class Tail(Component):
	def __init__(self, horzStab, vertStab):

		self.horzStab = horzStab
		self.vertStab = vertStab

		self._weight = horzStab.weight + vertStab.weight

		pass

	@property
	def weight(self):
		return self._weight
