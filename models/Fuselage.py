from .Component import Component

# Fuselage consists of three sections: Nose, CentreFuselage, Afterbody.

class Fuselage(Component):
	def __init__(self, centreFuselage, afterbody, nose):

		self.centreFuselage = centreFuselage
		self.afterbody = afterbody
		self.nose = nose
