class Component():

    def __init__(self):

        pass

    def __repr__(self):
        out = f'\n Object: {self.__class__.__name__}'

        for k in self.__dict__:
            val = str(self.__dict__[k])
            val = val.replace('\n', '\n --> ')
            out += f'\n{k} \t {val}'

        return out
