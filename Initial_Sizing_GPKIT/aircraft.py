"""Modular aircraft concept"""
import pickle
import numpy as np
from gpkit import Model, Variable, Vectorize, parse_variables

class Mission(Model):
    """A sequence of flight segments

    Upper Unbounded
    ---------------
    aircraft.wing.c, aircraft.wing.A

    Lower Unbounded
    ---------------
    aircraft.W
    """
    def setup(self, aircraft):
        self.aircraft = aircraft

        with Vectorize(4):  # four flight segments
            self.fs = FlightSegment(aircraft)

        Wburn = self.fs.aircraftp.Wburn
        Wfuel = self.fs.aircraftp.Wfuel
        self.takeoff_fuel = Wfuel[0]

        return {
            "definition of Wburn":
                Wfuel[:-1] >= Wfuel[1:] + Wburn[:-1],
            "require fuel for the last leg":
                Wfuel[-1] >= Wburn[-1],
            "flight segment":
                self.fs}



class FlightSegment(Model):
    """Combines a context (flight state) and a component (the aircraft)

    Upper Unbounded
    ---------------
    Wburn, aircraft.wing.c, aircraft.wing.A

    Lower Unbounded
    ---------------
    Wfuel, aircraft.W

    """
    def setup(self, aircraft):
        self.aircraft = aircraft

        self.flightstate = FlightState()
        self.aircraftp = aircraft.dynamic(aircraft, self.flightstate)

        self.Wburn = self.aircraftp.Wburn
        self.Wfuel = self.aircraftp.Wfuel

        return {"flightstate": self.flightstate,
                "aircraft performance": self.aircraftp}

class FlightState(Model):
    """Context for evaluating flight physics

    Variables
    ---------
    V     40       [knots]    true airspeed
    mu    1.628e-5 [N*s/m^2]  dynamic viscosity
    rho   0.74     [kg/m^3]   air density

    """
    #@parse_variables(__doc__, globals())
    def setup(self):

        V = self.V = Variable("V", 40, "knots", "true airspeed")
        mu = self.mu = Variable("mu", 1.628e-5, "N*s/m^2", "dynamic viscosity")
        rho = self.rho = Variable("rho", 0.74, "kg/m^3", "air density")
        pass




class AircraftP(Model):
    """Aircraft flight physics: weight <= lift, fuel burn

    Variables
    ---------
    Wfuel  [lbf]  fuel weight
    Wburn  [lbf]  segment fuel burn

    Upper Unbounded
    ---------------
    Wburn, aircraft.wing.c, aircraft.wing.A

    Lower Unbounded
    ---------------
    Wfuel, aircraft.W, state.mu

    """
    #@parse_variables(__doc__, globals())
    def setup(self, aircraft, state):

        Wfuel = self.Wfuel = Variable("Wfuel", "lbf", "Fuel weight")
        Wburn = self.Wburn = Variable("Wburn", "lbf", "Segment fuel burn")

        self.aircraft = aircraft
        self.state = state

        self.wing_aero = aircraft.wing.dynamic(aircraft.wing, state)
        self.perf_models = [self.wing_aero]

        W = aircraft.W
        S = aircraft.wing.S

        V = state.V
        rho = state.rho

        D = self.wing_aero.D
        CL = self.wing_aero.CL

        return {
            "lift":
                W + Wfuel <= 0.5*rho*CL*S*V**2,
            "fuel burn rate":
                Wburn >= 0.1*D,
            "performance":
                self.perf_models}



class Aircraft(Model):
    """The vehicle model

    Variables
    ---------
    W  [lbf]  weight

    Upper Unbounded
    ---------------
    W

    Lower Unbounded
    ---------------
    wing.c, wing.S
    """
    #@parse_variables(__doc__, globals())
    def setup(self):

        W = self.W = Variable("W", "lbf", "Aircraft weight")

        self.fuse = Fuselage()
        self.wing = Wing()
        self.components = [self.fuse, self.wing]

        return {
            "definition of W":
                W >= sum(c.W for c in self.components),
            "components":
                self.components}

    dynamic = AircraftP



class WingAero(Model):
    """Wing aerodynamics

    Variables
    ---------
    CD      [-]    drag coefficient
    CL      [-]    lift coefficient
    e   0.9 [-]    Oswald efficiency
    Re      [-]    Reynold's number
    D       [lbf]  drag force

    Upper Unbounded
    ---------------
    D, Re, wing.A, state.mu

    Lower Unbounded
    ---------------
    CL, wing.S, state.mu, state.rho, state.V
    """
    #@parse_variables(__doc__, globals())
    def setup(self, wing, state):

        CD = self.CD = Variable("CD", "", "drag coefficient")
        CL = self.CL = Variable("CL", "", "lift coefficient")
        e  = self.e  = Variable("e", 0.9, "", "Oswald efficiency")
        Re = self.Re = Variable("Re", "", "Reynold's number")
        D  = self.D  = Variable("D", "lbf", "Drag force")

        self.wing = wing
        self.state = state

        c = wing.c
        A = wing.A
        S = wing.S
        rho = state.rho
        V = state.V
        mu = state.mu

        return {
            "drag model":
                CD >= 0.074/Re**0.2 + CL**2/(np.pi*A*e),
            "definition of Re":
                Re == rho*V*c/mu,
            "definition of D":
                D >= 0.5*rho*V**2*CD*S}


class Wing(Model):
    """Aircraft wing model

    Variables
    ---------
    W        [lbf]       weight
    S        [ft^2]      surface area
    rho    1 [lbf/ft^2]  areal density
    A     27 [-]         aspect ratio
    c        [ft]        mean chord

    Upper Unbounded
    ---------------
    W

    Lower Unbounded
    ---------------
    c, S
    """
    #@parse_variables(__doc__, globals())
    def setup(self):

        W = self.W = Variable("W", "lbf", "weight")
        S = self.S = Variable("S", "ft^2", "surface area")
        rho = self.rho = Variable("rho", 1, "lbf/ft^2", "areal density")
        A = self.A = Variable("AR", 27, "", "aspect ratio")
        c = self.c = Variable("c", "ft", "mean chord")

        return {"parametrization of wing weight":
                    W >= S*rho,
                "definition of mean chord":
                    c == (S/A)**0.5}

    dynamic = WingAero


class Fuselage(Model):
    """The thing that carries the fuel, engine, and payload

    A full model is left as an exercise for the reader.

    Variables
    ---------
    W  100 [lbf]  weight

    """
    #@parse_variables(__doc__, globals())
    def setup(self):
        W = self.W = Variable("W", 100, "lbf", "Weight of Fuselage")
        pass
