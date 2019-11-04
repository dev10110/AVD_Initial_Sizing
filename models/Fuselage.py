# Fuselage consists of three sections: Nose, CentreFuselage, Afterbody.

class Fuselage:
	def __init__(self, centreFuselage, afterbody, nose):
		self.centreFuselage = centreFuselage
		self.afterbody = afterbody
		self.nose = nose

	def __repr__(self):
		out = f'\n Object: {self.__class__.__name__}'

		for k in self.__dict__:
			out += f'\n{k} \t {self.__dict__[k]}'

		return out
