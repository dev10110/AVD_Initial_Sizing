import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib import cm
import matplotlib as mpl
import pandas as pd
import scipy as sp
from scipy.optimize import NonlinearConstraint, Bounds
import warnings


cmap  = cm.get_cmap('RdBu', 100)
cmap2 = cm.get_cmap('Blues', 100)


class Node:
    def __init__(self, x, y, freex=True, freey=True):

        self.x = x
        self.y = y

        self.freex = freex
        self.freey = freey

        self.dx = 0
        self.dy = 0

        self.fx = 0
        self.fy = 0

    def pos(self):

        return np.array([self.x, self.y])

    def __eq__(self, other):
        # only checks for position, not for whether the boundarys are defined the same
        return np.allclose(self.pos(), other.pos())

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f'N({self.x:2.2f}, {self.y:2.2f})'

    def plot(self):
        if self.freex and self.freey:
            plt.plot(self.x, self.y, 'ko')
        elif self.freex and not self.freey:
            plt.plot(self.x, self.y, 'bo')
        elif not self.freex and self.freey:
            plt.plot(self.x, self.y, 'go')
        else:
            plt.plot(self.x, self.y, 'ro')

    def apply_load(self, fx, fy):
        # sets the load of a node to a value
        self.fx = fx
        self.fy = fy

    def add_load(self, fx, fy):
        # adds the load to the previously stored value
        self.fx += fx
        self.fy += fy


class Bar:

    def __init__(self, node0, node1, w=5e-3, t=6.35e-3, E=71e9, yield_strength=300e6, density=2700):

        if node0 == node1:
            raise ValueError(
                "Node 0 and Node 1 cannot be the same. Check if they are at the same place.")

        self.node0 = node0
        self.node1 = node1
        self.w = w
        self.t = t
        self.E = E
        self.yield_strength = yield_strength
        self.density = density

    def __eq__(self, other):
        # only checks if the same nodes are used, not for material or thicknesses.

        if (self.node0 == other.node0 and self.node1 == other.node1) or (self.node0 == other.node1 and self.node1 == other.node0):
            return True
        else:
            return False

    def __hash__(self):
        n0 = self.node0
        n1 = self.node1

        if n0.x < n1.x:
            return hash((n0, n1))
        if n0.x > n1.x:
            return hash((n1, n0))
        if n0.y < n1.y:
            return hash((n0, n1))
        if n0.y > n1.y:
            return hash((n1, n0))

        raise RuntimeError("Check if the two nodes of this bar are at the same place.")

    def __repr__(self):
        return f'B({self.node0}, {self.node1})'

    def length(self):
        """Get length of vector between n0 and n1"""

        v = self.node1.pos() - self.node0.pos()

        return np.sqrt(np.dot(v, v))

    def e(self): # def e vector
        th = self.theta()
        return np.array([np.cos(th), np.sin(th)])

    def area(self):
        return self.w * self.t

    def volume(self):

        return self.area()*self.length()

    def mass(self):

        return self.volume()*self.density

    def I(self):

        b = max(self.w, self.t)
        h = min(self.w, self.t)
        # can buckle in either direction, so need to compute both Is and take the smaller None
        I = b*h**3/12

        return I

    def EA(self):
        return self.E * self.area()

    def stiffness(self):

        eeT = np.outer(self.e(), self.e())

        return (self.EA()/self.length())*np.block([[eeT, -eeT], [-eeT, eeT]])

    def theta(self):
        """Get angle of vector from n0 to n1"""

        v = self.node1.pos() - self.node0.pos()

        return np.arctan2(v[1], v[0])

    def extension(self):

        du1 = np.array([self.node0.dx, self.node0.dy])
        du2 = np.array([self.node1.dx, self.node1.dy])

        return np.dot((du2-du1), self.e())

    def strain(self):

        delta = self.extension()

        return delta/self.length()

    def stress(self):

        strain = self.strain()

        return self.E * strain

    def tension(self):

        return self.stress() * self.area()

    def buckling_load(self):

        Fcrit = np.pi**2*self.E*self.I()/self.length()**2

        return Fcrit

    def qBuckle(self):

        if self.tension() >= 0.0:
            return False

        if abs(self.tension()) > self.buckling_load():
            return True
        else:
            return False

    def qYield(self):

        if self.stress() <= 0.0:
            return False
        if self.stress() > self.yield_strength:
            return True
        else:
            return False


    def plot(self, color='k', def_scale=1.0):
        nodes = [self.node0, self.node1]

        x = [n.x for n in nodes]
        y = [n.y for n in nodes]

        xdx = [n.x + def_scale*n.dx for n in nodes]
        ydy = [n.y + def_scale*n.dy for n in nodes]

        plt.plot(x, y, '0.8')
        plt.plot(xdx, ydy, color=color)

class Truss:
    def __init__(self, bars):
        """Class to define a truss"""

        # remove duplicated bars
        self.bars = list(set(bars))

        # extract nodes
        self.nodes = self.extract_nodes(bars)

    @classmethod
    def from_delaunay(cls, nodes):

        def find_neighbors(x, triang): return list(
            set(indx for simplex in triang.simplices if x in simplex for indx in simplex if indx != x))

        # create numpy nodes array
        node_points = np.vstack([n.pos() for n in nodes])

        # perform the triangulation
        d = Delaunay(node_points)

        bars = []

        for i, node in enumerate(nodes):

            neighbors = find_neighbors(i, d)
            for n in neighbors:
                bars.extend([Bar(node, nodes[n]) for n in neighbors])

        # note, there will be repeated bars here, but the clean up when creating the bars will fix this issue.
        return cls(bars)

    @classmethod
    def from_fully_connected(cls, nodes):

        bars = set()

        for node0 in nodes:
            for node1 in nodes:
                if not node0 == node1:
                    bars.add(Bar(node0, node1))

        return cls(bars)

    def solve(self, method="solve"):
        # create stiffness matrix (we will delete the non-free nodes at the end)
        # for each node, assemble the stiffness matrix
        # 2*nodes because each node can have 2 degrees of freedom
        nodes = self.nodes
        bars = self.bars

        stiffness = np.zeros([2*len(nodes), 2*len(nodes)])

        for bar in self.bars:

            n0ind = nodes.index(bar.node0)
            n1ind = nodes.index(bar.node1)

            # replace all 16 elements
            bstiff = bar.stiffness()
            stiffness[2*n0ind:2*n0ind+2, 2*n0ind:2*n0ind+2] += bstiff[0:2, 0:2]
            stiffness[2*n0ind:2*n0ind+2, 2*n1ind:2*n1ind+2] += bstiff[0:2, 2:4]
            stiffness[2*n1ind:2*n1ind+2, 2*n0ind:2*n0ind+2] += bstiff[2:4, 0:2]
            stiffness[2*n1ind:2*n1ind+2, 2*n1ind:2*n1ind+2] += bstiff[2:4, 2:4]

        # create force vector
        F = np.zeros(2*len(self.nodes))
        for i, node in enumerate(self.nodes):
            F[2*i] = node.fx
            F[2*i+1] = node.fy

        # list the rows that are with fixed constraints
        delrows = []
        for i, node in enumerate(self.nodes):
            if node.freex == False:
                delrows.append(2*i)
            if node.freey == False:
                delrows.append(2*i+1)

        # delete rows and cols from the stiffness matrix
        stiffness = np.delete(stiffness, delrows, axis=0)
        stiffness = np.delete(stiffness, delrows, axis=1)

        # store into the stiffness of the truss
        self.stiffness = stiffness

        # delete rows from the force vector
        F = np.delete(F, delrows)
        # store into a force vector
        self.F = F

        # try to solve stiffness*deflections = forces for deflections
        try:
            if method == "solve":
                u = np.linalg.solve(stiffness, F)

            elif method == "lstsq":
                sol = np.linalg.lstsq(stiffness, F)
                self.lstsq_sol = sol
                u = sol[0]

        except Exception as e:
            raise RuntimeError(f"Oops. Solve failed. \n {e}")

        self.u = u

        # reconstruct deformations
        ind = 0
        for i, node in enumerate(self.nodes):
            if node.freex:
                node.dx = u[ind]
                ind += 1
            # else skip the node y deflection
            if node.freey:
                node.dy = u[ind]
                ind += 1
            # else skip the node y deflection

        # does not return anything
        return None

    def plot(self,def_scale=1.0, figsize=(12, 5)):

        plt.figure(figsize=figsize)

        plt.subplot(121)

        self.plot_tensions(def_scale=def_scale)

        plt.subplot(122)

        self.plot_stress(def_scale=def_scale)


    def plot_tensions(self, ax=None, def_scale=1.0):

        if ax is None:
            ax = plt.gca()

        # create a symmetric stress bar
        Trange = max(abs(b.tension()) for b in self.bars)

        # plot all bars
        for bar in self.bars:

            if Trange < 0.1:
                c = '0.8'
            else:
                c = cmap((bar.tension()+Trange)/(2*Trange))

            bar.plot(color=c,def_scale=def_scale)
        for node in self.nodes:
            node.plot()

        self.plot_force_quiver(ax)

        # finally put in the colorbar:
        (cax, kw) = mpl.colorbar.make_axes(ax)
        norm = mpl.colors.Normalize(vmin=-Trange, vmax=+Trange)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_label('Bar Tensions (N)')

    def plot_force_quiver(self, ax):

        # plot the arrows
        x = [n.x for n in self.nodes if not (n.fx == 0 and n.fy == 0)]
        y = [n.y for n in self.nodes if not (n.fx == 0 and n.fy == 0)]
        fx = [n.fx for n in self.nodes if not (n.fx == 0 and n.fy == 0)]
        fy = [n.fy for n in self.nodes if not (n.fx == 0 and n.fy == 0)]

        ax.quiver(x, y, fx, fy, color='red',zorder=10)

    def plot_stress(self, ax=None, def_scale=1.0):

        if ax is None:
            ax = plt.gca()

        # create a symmetric stress bar
        Srange = max(abs(b.stress()) for b in self.bars)

        # plot all bars
        for bar in self.bars:

            if Srange < 0.1:
                c = '0.8'
            else:
                c = cmap((bar.stress()+Srange)/(2*Srange))

            bar.plot(color=c,def_scale=def_scale)
        for node in self.nodes:
            node.plot()

        # plot the arrows
        self.plot_force_quiver(ax)


        # finally put in the colorbar:
        (cax, kw) = mpl.colorbar.make_axes(ax)
        norm = mpl.colors.Normalize(vmin=-Srange, vmax=+Srange)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_label('Bar Stress (Pa)')



    def plot_widths(self, ax=None, def_scale=1.0):

        if ax is None:
            ax = plt.gca()

        # create a symmetric stress bar
        Wrange = 1000*max(b.w for b in self.bars)

        # plot all bars
        for bar in self.bars:

            if Wrange < 0.01e-3:
                c = '0.95'
            else:
                c = cmap2(bar.w*1000/Wrange)

            bar.plot(color=c,def_scale=def_scale)
        for node in self.nodes:
            node.plot()

        # plot the arrows
        self.plot_force_quiver(ax)


        # finally put in the colorbar:
        (cax, kw) = mpl.colorbar.make_axes(ax)
        norm = mpl.colors.Normalize(vmin=0, vmax=Wrange)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap2, norm=norm)
        cb.set_label('Bar Width (mm)')


    def details(self):

        # details on each node:
        # node num | x | y | freex? | freey? | fx | fy || dx | dy |

        node_head = ["ID", "x (m)", "y (m)", "Free x?", "Free y?", "Force x (N)", "Force y (N)", "Delta x (mm)", "Delta y (mm)"]

        node_deets = [[i, n.x, n.y, n.freex, n.freey, n.fx, n.fy, n.dx*1000, n.dy*1000] for i, n in enumerate(self.nodes)]

        # details for each bar:
        # bar num | node0num (x,y) | node1num (x,y)| E | w | t | L || T | ext | stress | strain

        bar_head = ["ID", "Node 0","Node 1", "E (GPa)",  "Yield (MPa)", "w (mm)", "t (mm)", "A (mm2)", "I (mm4)", "L (m)", "m (kg)", "Buckling Load (N)", "T (N)", "ext (mm)", "Stress (MPa)", "Strain", "Will buckle?", "Will yield?", "Buckle Margin", "Yield Margin"]

        bar_deets = [[i, b.node0, b.node1, b.E/10**9, b.yield_strength/10**6, b.w*1000, b.t*1000, b.area()*10**6, b.I()*10**12, b.length(), b.mass(), b.buckling_load(), b.tension(), b.extension()*1000, b.stress()/10**6, b.strain(), b.qBuckle(), b.qYield(), -min(b.buckling_load()/b.tension(), 0), b.yield_strength/b.stress()] for i, b in enumerate(self.bars)]

        df_nodes = pd.DataFrame(node_deets, columns=node_head)

        df_bars = pd.DataFrame(bar_deets, columns=bar_head)

        return df_nodes, df_bars


    def mass(self):

        return sum(bar.mass() for bar in self.bars)


    def extract_nodes(self, bars):

        nodes = set()

        for bar in bars:
            nodes.add(bar.node0)
            nodes.add(bar.node1)

        return list(nodes)

    def tensions(self):
        return [bar.tension() for bar in self.bars]

    def extensions(self):
        return [bar.extension() for bar in self.bars]

    def set_widths(self, widths):

        for i, bar in enumerate(self.bars):
            bar.w = widths[i]

    def set_all_widths(self, width):
        for b in self.bars:
            b.w = width


    def miminize_mass(self, deflection_constraints=None, extra_constraints=None, buckling_SF=1.5, yield_SF=1.5, keep_feasible=False, method='SLSQP', **kwargs):

        def f_objective(widths):

            self.set_widths(widths)

            return self.mass()


        def f_deflection_con(widths):

            self.set_widths(widths)

            self.solve()

            con = []

            for defl in deflection_constraints:
                node, dx_min, dx_max, dy_min, dy_max = defl

                if dx_min is not None:
                    con.append(node.dx - dx_min)
                if dx_max is not None:
                    con.append(dx_max - node.dx)
                if dy_min is not None:
                    con.append(node.dy - dy_min)
                if dy_max is not None:
                    con.append(dy_max - node.dy)

            return con

        def f_buckling_con(widths):
            self.set_widths(widths)

            self.solve()

            return [buckling_SF*bar.tension() + bar.buckling_load() for bar in self.bars]

        def f_yield_con(widths):
            self.set_widths(widths)

            self.solve()

            return [bar.yield_strength - yield_SF*bar.stress() for bar in self.bars]

        buckling_constraint = NonlinearConstraint(f_buckling_con, lb=0, ub=np.inf, keep_feasible=keep_feasible)
        yield_constraint = NonlinearConstraint(f_yield_con, lb=0, ub=np.inf, keep_feasible=keep_feasible)
        constraints = [buckling_constraint, yield_constraint]


        if deflection_constraints is not None:
            deflection_constraint = NonlinearConstraint(f_deflection_con, lb=0, ub=np.inf, keep_feasible=keep_feasible)
            constraints.append(deflection_constraint)

        if extra_constraints is not None:
            constraints.extend(extra_constraints)

        minW = 0.1e-3;
        maxW = 20e-3;

        widths0 = [bar.w for bar in self.bars]

        bounds = Bounds(minW, maxW, keep_feasible=keep_feasible)

        sol = sp.optimize.minimize(f_objective, x0=widths0, bounds=bounds, constraints=constraints, method=method, **kwargs)

        if not sol.success:
            warnings.warn("Optimization hasn't been successful! Please check!")

        return sol
