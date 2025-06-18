from sympy import Symbol, Function, sqrt, Number, Min
from modulus.geometry import Parameterization
from modulus.eq.pde import PDE

class ZeroEquation(PDE):
    """
    Zero Equation Turbulence model

    Parameters
    ==========
    nu : float

        The kinematic viscosity of the fluid.
    max_distance : float

        The maximum wall distance in the flow field.
    rho : float, Sympy Symbol/Expr, str

        The density. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.

        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation. Default is 1.
    dim : int
        Dimension of the Zero Equation Turbulence model (2 or 3).
        Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.

    Example
    ========
    >>> zeroEq = ZeroEquation(nu=0.1, max_distance=2.0, dim=2)
    >>> kEp.pprint()
      nu: sqrt((u__y + v__x)**2 + 2*u__x**2 + 2*v__y**2)
      *Min(0.18, 0.419*normal_distance)**2 + 0.1
    """

    name = "ZeroEquation"

    def __init__(
        self, nu, max_distance, rho=1, dim=3, time=True, parameterization=Parameterization()
    ):  # TODO add density into model
        # set params
        self.dim = dim
        self.time = time

        # model coefficients
        self.max_distance = max_distance
        self.karman_constant = 0.41
        self.max_distance_ratio = 0.09

        # coordinates
        x, y, z, r_ratio = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("r_ratio")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t, "r_ratio":r_ratio}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # density
        if type(rho) is str:
            rho = Function(rho)(*input_variables)
        elif type(rho) in [float, int]:
            rho = Number(rho)

        # wall distance
        normal_distance = Function("sdf")(*input_variables)

        # mixing length
        mixing_length = Min(
            self.karman_constant * normal_distance,
            self.max_distance_ratio * self.max_distance,
        )
        mixing_length = self.max_distance_ratio * self.max_distance
        G = (
            2 * u.diff(x) ** 2
            + 2 * v.diff(y) ** 2
            + 2 * w.diff(z) ** 2
            + (u.diff(y) + v.diff(x)) ** 2
            + (u.diff(z) + w.diff(x)) ** 2
            + (v.diff(z) + w.diff(y)) ** 2
        )

        # set equations
        self.equations = {}
        self.equations["nu"] = nu +  mixing_length**2 *sqrt(G)

class NavierStokes(PDE):
    """
    Compressible Navier Stokes equations

    Parameters
    ==========
    nu : float, Sympy Symbol/Expr, str
        The kinematic viscosity. If `nu` is a str then it is
        converted to Sympy Function of form `nu(x,y,z,t)`.
        If `nu` is a Sympy Symbol or Expression then this
        is substituted into the equation. This allows for
        variable viscosity.
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressible Navier Stokes. Default is 1.
    dim : int
        Dimension of the Navier Stokes (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the Navier-Stokes equations.

    Examples
    ========
    >>> ns = NavierStokes(nu=0.01, rho=1, dim=2)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: u*u__x + v*u__y + p__x + u__t - 0.01*u__x__x - 0.01*u__y__y
      momentum_y: u*v__x + v*v__y + p__y + v__t - 0.01*v__x__x - 0.01*v__y__y
    >>> ns = NavierStokes(nu='nu', rho=1, dim=2, time=False)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: -nu*u__x__x - nu*u__y__y + u*u__x + v*u__y - nu__x*u__x - nu__y*u__y + p__x
      momentum_y: -nu*v__x__x - nu*v__y__y + u*v__x + v*v__y - nu__x*v__x - nu__y*v__y + p__y
    """

    name = "NavierStokes"

    def __init__(self, nu, omega, rho=1, dim=3, time=True, mixed_form=False):
        # set params
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # pressure
        p = Function("p")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        # density
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # omega
        if isinstance(omega, str):
            omega = Function(omega)(*input_variables)
        elif isinstance(omega, (float, int)):
            omega = Number(omega)

        # dynamic viscosity
        rho2 = rho
        rho = rho
        mu = rho * nu

        u_r = u + y * omega
        v_r = v - x * omega

        S_MRF_x = -rho * (-v * omega)
        S_MRF_y = -rho * (u * omega)

        # set equations
        self.equations = {}
        self.equations["continuity"] = (
                rho2.diff(t) + (rho2 * u).diff(x) + (rho2 * v).diff(y) + (rho2 * w).diff(z)
        )

        self.equations["continuity_r"] = (

            (rho2.diff(t) + (rho2 * u_r).diff(x) + (rho2 * v_r).diff(y) + (rho2 * w).diff(z))
        )

        curl = Number(0) if rho.diff(x) == 0 else u.diff(x) + v.diff(y) + w.diff(z)
        self.equations["momentum_x"] = (
                (
                        (rho * u).diff(t)
                        + (
                                u * ((rho * u).diff(x))
                                + v * ((rho * u).diff(y))
                                + w * ((rho * u).diff(z))
                                + rho * u * (curl)
                        )
                        + p.diff(x)
                        - (-2 / 3 * mu * (curl)).diff(x)
                        - (mu * u.diff(x)).diff(x)
                        - (mu * u.diff(y)).diff(y)
                        - (mu * u.diff(z)).diff(z)
                        - (mu * (curl).diff(x))
                ) / 1
        )

        self.equations["momentum_x_r"] = (

                (
                        (rho * u_r).diff(t)
                        + (
                                u_r * ((rho * u).diff(x))
                                + v_r * ((rho * u).diff(y))
                                + w * ((rho * u).diff(z))
                            #     + u * ((rho * u_r).diff(x))
                            #    + v * ((rho * u_r).diff(y))
                            #   + w * ((rho * u_r).diff(z))
                        )
                        + p.diff(x)
                        - (-2 / 3 * mu * (curl)).diff(x)
                        - (mu * u.diff(x)).diff(x)
                        - (mu * u.diff(y)).diff(y)
                        - (mu * u.diff(z)).diff(z)
                        - (mu * (curl).diff(x))
                        - S_MRF_x
                ) / 1

        )
        self.equations["momentum_y"] = (
                (
                        (rho * v).diff(t)
                        + (
                                u * ((rho * v).diff(x))
                                + v * ((rho * v).diff(y))
                                + w * ((rho * v).diff(z))
                                + rho * v * (curl)
                        )
                        + p.diff(y)
                        - (-2 / 3 * mu * (curl)).diff(y)
                        - (mu * v.diff(x)).diff(x)
                        - (mu * v.diff(y)).diff(y)
                        - (mu * v.diff(z)).diff(z)
                        - (mu * (curl).diff(y))
                ) / 1
        )

        self.equations["momentum_y_r"] = (

                (
                        (rho * v_r).diff(t)
                        + (
                                u_r * ((rho * v).diff(x))
                                + v_r * ((rho * v).diff(y))
                                + w * ((rho * v).diff(z))
                            #               + u * ((rho * v_r).diff(x))
                            #              + v * ((rho * v_r).diff(y))
                            #             + w * ((rho * v_r).diff(z))
                        )
                        + p.diff(y)
                        - (-2 / 3 * mu * (curl)).diff(y)
                        - (mu * v.diff(x)).diff(x)
                        - (mu * v.diff(y)).diff(y)
                        - (mu * v.diff(z)).diff(z)
                        - (mu * (curl).diff(y))
                        - S_MRF_y
                ) / 1

        )
        self.equations["momentum_z"] = (
                (
                        (rho * w).diff(t)
                        + (
                                u * ((rho * w).diff(x))
                                + v * ((rho * w).diff(y))
                                + w * ((rho * w).diff(z))
                                + rho * w * (curl)
                        )
                        + p.diff(z)
                        - (-2 / 3 * mu * (curl)).diff(z)
                        - (mu * w.diff(x)).diff(x)
                        - (mu * w.diff(y)).diff(y)
                        - (mu * w.diff(z)).diff(z)
                        - (mu * (curl).diff(z))
                ) / 1
        )

        self.equations["momentum_z_r"] = (

                (
                        (rho * w).diff(t)
                        + (
                                u_r * ((rho * w).diff(x))
                                + v_r * ((rho * w).diff(y))
                                + w * ((rho * w).diff(z))
                                + rho * w * (curl)
                        )
                        + p.diff(z)
                        - (-2 / 3 * mu * (curl)).diff(z)
                        - (mu * w.diff(x)).diff(x)
                        - (mu * w.diff(y)).diff(y)
                        - (mu * w.diff(z)).diff(z)
                        - (mu * (curl).diff(z))
                ) / 1

        )