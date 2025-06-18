from sympy import And, Or
import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from custom_equation import NavierStokes, ZeroEquation
from modulus.eq.pdes.basic import NormalDotVec
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from modulus.domain.validator import PointwiseValidator
from modulus.utils.io import csv_to_dict
from modulus.key import Key

from geometry import *
from custom_plotter import *

@modulus.main(config_path="conf", config_name="config_flow.yaml")
def run(cfg: ModulusConfig) -> None:
    plt.switch_backend('agg')

    # physical quantities
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    h = Symbol('h')
    r = Symbol('r')

    radius = 0.05 + 0.02 * r
    height = 0.045 * h
    stirring_rate = 10.0 * str

    nu = 0.0000138 * 3  # Viscosity (kg/m*s)
    rho = 1  # Density (kg/m^3)

    inlet_u = 0.0  # m/s
    inlet_v = 0.0  # m/s

    inlet_w = -0.5 * (0.0004 - (x + 0.05) ** 2 - y ** 2) / 0.0004  # m/s
    volumetric_flow = 0.02 * 0.02 * 3.141592 * (0.5 / 2)

    noslip_u = 0.0  # m/s
    noslip_v = 0.0  # m/s
    noslip_w = 0.0  # m/s

    geo = Reactor(compartment=False)

    ze = ZeroEquation(nu=nu, dim=3, time=False, max_distance=0.07, parameterization=geo.pr)
    ns = NavierStokes(nu=ze.equations["nu"], omega=stirring_rate, rho=rho, dim=3, time=False)
    normal_dot_vel = NormalDotVec(["u", "v", "w"])

    equation_nodes = ns.make_nodes() + normal_dot_vel.make_nodes() + ze.make_nodes()

    flow_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("r"), Key("h"), Key("str")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        frequencies=("axis", [i for i in range(20)]),
        frequencies_params=("axis", [i for i in range(20)]),
        cfg=cfg.arch.modified_fourier,
    )

    flow_nodes = equation_nodes + [flow_net.make_node(name="flow_network")]

    # make domain
    domain = Domain()

    # inlet
    inlet1 = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.inlet_plane,
        outvar={"u": inlet_u, "v": inlet_v, "w": inlet_w},
        batch_size=100,
        criteria=And((x + 0.05) ** 2 + y ** 2 < 0.02 * 0.02, z > 0.099),
        lambda_weighting={"u": 1.0, "v": 1.0, "w": 1.0},
        batch_per_epoch=100,
        parameterization=geo.pr,
    )
    domain.add_constraint(inlet1, "inlet1")

    inlet2 = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.inlet_plane,
        outvar={"u": noslip_u, "v": noslip_v, "w": noslip_w},
        batch_size=100,
        criteria=And((x + 0.05) ** 2 + y ** 2 > 0.02 * 0.02, x ** 2 + y ** 2 < 0.1 * 0.1),
        lambda_weighting={"u": 1.0, "v": 1.0, "w": 1.0},
        batch_per_epoch=100,
        parameterization=geo.pr,
    )
    domain.add_constraint(inlet2, "inlet2")

    # outlet
    outlet1 = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.outlet_plane,
        outvar={"p": 0},
        batch_size=100,
        criteria=(x ** 2 + y ** 2 < 0.02 * 0.02),
        lambda_weighting={"p": 10.0},
        batch_per_epoch=100,
        parameterization=geo.pr,
    )
    domain.add_constraint(outlet1, "outlet1")

    outlet2 = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.outlet_plane,
        outvar={"u": 0, "v": 0, "w": 0},
        criteria=And(x ** 2 + y ** 2 > 0.02 * 0.02, x ** 2 + y ** 2 < 0.1 * 0.1),
        batch_size=100,
        lambda_weighting={"u": 1.0, "v": 1.0, "w": 1.0},
        batch_per_epoch=100,
        parameterization=geo.pr,
    )
    domain.add_constraint(outlet2, "outlet2")

    # no slip
    no_slip_1 = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.geo,
        outvar={"u": noslip_u, "v": noslip_v, "w": noslip_w},
        batch_size=200,
        criteria=And(z < 0.1, z > -0.1, x ** 2 + y ** 2 > 0.099 * 0.099),
        lambda_weighting={"u": 1.0, "v": 1.0, "w": 1.0},
        batch_per_epoch=200,
        parameterization=geo.pr,
    )
    domain.add_constraint(no_slip_1, "no_slip_1")

    no_slip_2 = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.geo,
        outvar={"u": -y * stirring_rate, "v": x * stirring_rate, "w": 0},
        batch_size=1000,
        criteria=And(z < height + 0.025, z > height - 0.025, x ** 2 + y ** 2 < (0.06 + r * 0.02) ** 2),
        lambda_weighting={"u": 1.0, "v": 1.0, "w": 1.0},
        batch_per_epoch=200,
        parameterization=geo.pr,
    )
    domain.add_constraint(no_slip_2, "no_slip_2")

    no_slip_3 = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.geo,
        outvar={"u": noslip_u, "v": noslip_v, "w": noslip_w},
        batch_size=500,
        criteria=And(z < 0.1, z > -0.0325, x ** 2 + y ** 2 < 0.0025 * 0.0025),
        lambda_weighting={"u": 1.0, "v": 1.0, "w": 1.0},
        batch_per_epoch=200,
        parameterization=geo.pr,
    )
    domain.add_constraint(no_slip_3, "no_slip_3")

    # interior constraints
    interior1 = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=geo.geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=2500,
        criteria=Or(Or(z > height + 0.025, z < height - 0.025),
                    And(z < height + 0.025, z > height - 0.025, x ** 2 + y ** 2 > (0.06 + r * 0.02) ** 2)),
        lambda_weighting={
            "continuity": Symbol("sdf"),
            "momentum_x": Symbol("sdf"),
            "momentum_y": Symbol("sdf"),
            "momentum_z": Symbol("sdf"),
        },
        compute_sdf_derivatives=True,
        batch_per_epoch=500,
        parameterization=geo.pr,
    )
    domain.add_constraint(interior1, "interior1")

    interior2 = PointwiseInteriorConstraint(
        nodes=flow_nodes,
        geometry=geo.geo,
        outvar={"continuity_r": 0, "momentum_x_r": 0, "momentum_y_r": 0, "momentum_z_r": 0},
        batch_size=500,
        criteria=And(z < height + 0.025, z > height - 0.025, x ** 2 + y ** 2 < (0.06 + r * 0.02) ** 2),
        lambda_weighting={
            "continuity_r": Symbol("sdf"),
            "momentum_x_r": Symbol("sdf"),
            "momentum_y_r": Symbol("sdf"),
            "momentum_z_r": Symbol("sdf"),
        },
        compute_sdf_derivatives=True,
        batch_per_epoch=500,
        parameterization=geo.pr,
    )
    domain.add_constraint(interior2, "interior2")

    integral_continuity_inlet = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.inlet_plane,
        outvar={"normal_dot_vel": volumetric_flow},
        batch_size=8,
        integral_batch_size=3000,
        criteria=((x + 0.05) ** 2 + y ** 2 < 0.02 * 0.02),
        lambda_weighting={"normal_dot_vel": 5},
        fixed_dataset=False,
        num_workers=4,
        parameterization=geo.pr,
    )
    domain.add_constraint(integral_continuity_inlet, "integral_continuity_inlet")

    integral_continuity_outlet = IntegralBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.outlet_plane,
        outvar={"normal_dot_vel": -volumetric_flow},
        batch_size=8,
        integral_batch_size=3000,
        criteria=(x ** 2 + y ** 2 < 0.02 * 0.02),
        lambda_weighting={"normal_dot_vel": 5},
        fixed_dataset=False,
        num_workers=4,
        parameterization=geo.pr,
    )
    domain.add_constraint(integral_continuity_outlet, "integral_continuity_outlet")

    mapping = {
        "x": "x",
        "y": "y",
        "z": "z",
        "x-velocity": "u",
        "y-velocity": "v",
        "z-velocity": "w",
        "pressure": "p",
    }

    fluent_var = csv_to_dict(
        to_absolute_path("fluent/001.csv"), mapping
    )

    shape = fluent_var["x"].shape
    fluent_var["r"] = np.full(shape, 0)
    fluent_var["h"] = np.full(shape, 0)
    fluent_var["str"] = np.full(shape, 1)

    fluent_invar_numpy = {
        key: value
        for key, value in fluent_var.items()
        if key in ["x", "y", "z", "r", "h", "str"]
    }
    fluent_outvar_numpy = {
        key: value
        for key, value in fluent_var.items()
        if key in ["u", "v", "w"]
    }

    grid_inference = PointwiseValidator(
        nodes=flow_nodes,
        invar=fluent_invar_numpy,
        true_outvar=fluent_outvar_numpy,
        plotter=ValidatorPlotter(),
    )
    domain.add_validator(grid_inference, "val")

    slv = Solver(cfg, domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()