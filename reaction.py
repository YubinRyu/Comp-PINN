import os
import shutil
import modulus
import torch
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from custom_equation import NavierStokes, ZeroEquation
from modulus.eq.pdes.basic import NormalDotVec
from modulus.domain.constraint import PointwiseBoundaryConstraint

from modulus.domain.validator import PointwiseValidator
from modulus.domain.monitor import PointwiseMonitor
from modulus.utils.io import csv_to_dict
from modulus.key import Key
from custom_plotter import *

from scipy.integrate import solve_ivp
from geometry import *
import pandas as pd

@modulus.main(config_path="conf", config_name="config_reaction.yaml")
def compartmentalization(cfg: ModulusConfig) -> None:
    # physical quantities
    stirring_rate = 10.0 * str

    nu = 0.0000138 * 3  # Viscosity (kg/m*s)
    rho = 1  # Density (kg/m^3)

    geo = Reactor(compartment=True)

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

    # dummy for domain definition
    dummy = PointwiseBoundaryConstraint(
        nodes=flow_nodes,
        geometry=geo.inlet_plane,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=1,
        batch_per_epoch=1,
        parameterization=geo.pr,
    )
    domain.add_constraint(dummy, "dummy")

    inlet_flow = PointwiseMonitor(
        geo.inlet_plane.sample_boundary(10000),
        output_names=["p"],
        metrics={
            "inlet": lambda var: torch.sum(var["area"] * var["w"] * ((var['x'] + 0.05) ** 2 + var['y'] ** 2 < 0.02 * 0.02)),
        },
        nodes=flow_nodes,
    )
    domain.add_monitor(inlet_flow)

    outlet_flow = PointwiseMonitor(
        geo.outlet_plane.sample_boundary(10000),
        output_names=["p"],
        metrics={
            "outlet": lambda var: torch.sum(var["area"] * var["w"] * ((var['x']) ** 2 + var['y'] ** 2 < 0.02 * 0.02)),
        },
        nodes=flow_nodes,
    )
    domain.add_monitor(outlet_flow)

    # Compartmental Formulation
    specific_param_ranges = {
        r: 0,
        h: 0,
        str: 1,
    }

    # add metrics for front pressure
    plane_param_ranges = {
        **specific_param_ranges,
    }

    Radial_Flow = []
    for i, coord in enumerate(geo.Cor1):
        Radial_Flow.append(0)
        Radial_Flow[i] = PointwiseMonitor(
            geo.Flow_network[coord[0]][coord[1]].sample_boundary(1000, parameterization=plane_param_ranges,),
            output_names=["p"],
            metrics={
                "flow_radial" + repr(i): lambda var: torch.sum(
                    var["area"] * (var["u"] * var["x"] + var["v"] * var["y"]) / (
                                (var["x"] ** 2 + var["y"] ** 2) ** 0.5)),
            },
            nodes=flow_nodes,
        )
        domain.add_monitor(Radial_Flow[i])

    Axial_Flow = []
    for i, coord in enumerate(geo.Cor2):
        Axial_Flow.append(0)
        Axial_Flow[i] = PointwiseMonitor(
            geo.Flow_network[coord[0]][coord[1]].sample_boundary(1000, parameterization=plane_param_ranges,),
            output_names=["p"],
            metrics={
                "flow_axial" + repr(i): lambda var: torch.sum(var["area"] * var["w"]),
            },
            nodes=flow_nodes,
        )
        domain.add_monitor(Axial_Flow[i])

    Tangential_Flow = []
    for i, coord in enumerate(geo.Cor3):
        Tangential_Flow.append(0)
        Tangential_Flow[i] = PointwiseMonitor(
            geo.Flow_network[coord[0]][coord[1]].sample_boundary(1000, parameterization=plane_param_ranges,),
            output_names=["p"],
            metrics={
                "flow_tangential" + repr(i): lambda var: torch.sum(
                    var["area"] * (var["normal_x"] * var["u"] + var["normal_y"] * var["v"])),
            },
            nodes=flow_nodes,
        )
        domain.add_monitor(Tangential_Flow[i])

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
    slv.eval()

    import gc
    gc.collect()
    torch.cuda.empty_cache()

def polymerization():
    geo = Reactor(compartment=True)
    Initial_net = geo.Net

    for i, cor in enumerate(geo.Cor3):
        Initial_net[cor[0], cor[1]] = - \
            pd.read_csv('outputs/reaction/monitors/flow_tangential' + repr(i) + '.csv').iloc[0, 1]

    for i, cor in enumerate(geo.Cor2):
        Initial_net[cor[0], cor[1]] = \
            pd.read_csv('outputs/reaction/monitors/flow_axial' + repr(i) + '.csv').iloc[0, 1]

    for i, cor in enumerate(geo.Cor1):
        Initial_net[cor[0], cor[1]] = - \
            pd.read_csv('outputs/reaction/monitors/flow_radial' + repr(i) + '.csv').iloc[0, 1]

    Flow_network = (-Initial_net.T + Initial_net)

    Resi = np.sum(Flow_network, axis=1)
    volume = np.array(geo.volume)
    Compart_N = len(Resi)

    Flow_Network = Flow_network * 1000 - np.diag(Resi) * 1000  # [L]

    Volume = volume * 1000  # [L]
    N = Compart_N

    # Initial condition
    Initial_initiator = np.zeros(Compart_N)  # [mol/L]
    Initial_monomer = np.ones(Compart_N) * 18.57  # [mol/L]
    Initial_CTA = np.zeros(Compart_N)  # [mol/L]
    Initial_mu_0 = np.ones(Compart_N) * 0.001  # [mol/L]
    Initial_mu_1 = np.ones(Compart_N) * 0.001  # [mol/L]
    Initial_mu_2 = np.ones(Compart_N) * 0.001  # [mol/L]

    Initial = np.hstack((Initial_initiator, Initial_monomer, Initial_CTA, Initial_mu_0, Initial_mu_1, Initial_mu_2))
    t_span = (0, 500)

    def fun2(t, Initial, Flow_Network, Volume):
        Inlet_list = [51, 52]
        Side_inlet = 9
        N = len(Flow_Network)

        # Initialization
        I = np.clip(Initial[0 * N:1 * N], 0, None)  # [mol/L]
        M = np.clip(Initial[1 * N:2 * N], 0, None)  # [mol/L]
        CTA = np.clip(Initial[2 * N:3 * N], 0, None)  # [mol/L]
        Mu_0 = np.clip(Initial[3 * N:4 * N], 0, None)  # [mol/L]
        Mu_1 = np.clip(Initial[4 * N:5 * N], 0, None)  # [mol/L]
        Mu_2 = np.clip(Initial[5 * N:6 * N], 0, None)  # [mol/L]
        T = np.ones(N) * 450

        # Kinetics
        f = 0.55
        f_side = 0.9
        kd = 1.54e14 * np.exp(-15023 / T)
        kp = 1.25e8 * np.exp(-3800 / T)
        ktc = 1.25e9 * np.exp(-327 / T)
        ktd = 1.25e9 * np.exp(-327 / T)
        kctp = 4.38e8 * np.exp(-6603 / T)
        kcta = 3.44e6 * np.exp(-26800 / 8.3 / T)
        kctm = 8.7e5 * np.exp(-5032 / T)
        kbeta = 1.292e7 * np.exp(-5671 / T)

        # Algebraic part
        Mu_3 = np.clip((Mu_2 / (Mu_1 + 0.000001)) ** 2, 0, None) * np.clip(2 * Mu_0 * Mu_2 - Mu_1 ** 2, 0, None) ** 0.5
        L0 = abs((kd * I * f / ktc)) ** 0.5
        L0_side = abs((kd * I * f_side / ktc)) ** 0.5
        L0[9] = L0_side[9]
        L1 = abs(kp * M * L0 + kctm * L0 * M + kctp * L0 * Mu_2 + kbeta * L0) / abs(
            ktc * L0 + ktd * L0 + kctm * M + kcta * CTA + kctp * Mu_1 + kbeta)
        L2 = abs(kp * M * L0 + kp * M * 2 * L1 + kctm * L0 * M + kctp * L0 * Mu_3) / abs(
            ktc * L0 + ktd * L0 + kctm * M + kcta * CTA + kctp * Mu_1 + kbeta)

        # Integral part

        # Transport
        dIdt = (np.clip(Flow_Network, 0, None) @ I[:, np.newaxis]) / Volume[:, np.newaxis] \
               + (np.sum(np.clip(Flow_Network, None, 0), axis=1) * (I / Volume))[:, np.newaxis]
        dIdt = np.squeeze(dIdt)

        dMdt = (np.clip(Flow_Network, 0, None) @ M[:, np.newaxis]) / Volume[:, np.newaxis] \
               + (np.sum(np.clip(Flow_Network, None, 0), axis=1) * (M / Volume))[:, np.newaxis]
        dMdt = np.squeeze(dMdt)

        dCTAdt = (np.clip(Flow_Network, 0, None) @ CTA[:, np.newaxis]) / Volume[:, np.newaxis] \
                 + (np.sum(np.clip(Flow_Network, None, 0), axis=1) * (CTA / Volume))[:, np.newaxis]
        dCTAdt = np.squeeze(dCTAdt)

        dMu_0dt = (np.clip(Flow_Network, 0, None) @ Mu_0[:, np.newaxis]) / Volume[:, np.newaxis] \
                  + (np.sum(np.clip(Flow_Network, None, 0), axis=1) * (Mu_0 / Volume))[:, np.newaxis]
        dMu_0dt = np.squeeze(dMu_0dt)

        dMu_1dt = (np.clip(Flow_Network, 0, None) @ Mu_1[:, np.newaxis]) / Volume[:, np.newaxis] \
                  + (np.sum(np.clip(Flow_Network, None, 0), axis=1) * (Mu_1 / Volume))[:, np.newaxis]
        dMu_1dt = np.squeeze(dMu_1dt)

        dMu_2dt = (np.clip(Flow_Network, 0, None) @ Mu_2[:, np.newaxis]) / Volume[:, np.newaxis] \
                  + (np.sum(np.clip(Flow_Network, None, 0), axis=1) * (Mu_2 / Volume))[:, np.newaxis]
        dMu_2dt = np.squeeze(dMu_2dt)

        # Reaction part
        dIdt = dIdt - kd * I
        dMdt = dMdt - kp * L0 * M - kctm * L0 * M
        dCTAdt = dCTAdt - kctm * L0 * M
        dMu_0dt = dMu_0dt + 1 / 2 * ktc * L0 ** 2 + ktd * L0 ** 2 + kctm * M * L0 \
                  + kcta * L0 * CTA + kbeta * L0 * Mu_1
        dMu_1dt = dMu_1dt + (ktc + ktd) * L0 * L1 + kctm * M * L1 + kcta * L1 * CTA \
                  + kctp * (L1 * Mu_1 - L0 * Mu_2) + kbeta * (L1 * Mu_1 - L0 * Mu_2 / 2)
        dMu_2dt = dMu_2dt + (ktc + ktd) * L0 * L2 + ktc * L1 * L1 + kctm * M * L2 + kcta * L2 * CTA \
                  + kctp * (L2 * Mu_1 - L0 * Mu_3) + kbeta * (L2 * Mu_1 - 2 / 3 * Mu_3 * L0)

        # Inlet part
        for N_inlet in Inlet_list:
            dIdt[N_inlet] = dIdt[N_inlet] - Flow_Network[N_inlet, N_inlet] * I[N_inlet] / Volume[N_inlet]
            dMdt[N_inlet] = dMdt[N_inlet] - Flow_Network[N_inlet, N_inlet] * M[N_inlet] / Volume[N_inlet]
            dCTAdt[N_inlet] = dCTAdt[N_inlet] - Flow_Network[N_inlet, N_inlet] * CTA[N_inlet] / Volume[N_inlet]
            dMu_0dt[N_inlet] = dMu_0dt[N_inlet] - Flow_Network[N_inlet, N_inlet] * Mu_0[N_inlet] / Volume[N_inlet]
            dMu_1dt[N_inlet] = dMu_1dt[N_inlet] - Flow_Network[N_inlet, N_inlet] * Mu_1[N_inlet] / Volume[N_inlet]
            dMu_2dt[N_inlet] = dMu_2dt[N_inlet] - Flow_Network[N_inlet, N_inlet] * Mu_2[N_inlet] / Volume[N_inlet]

            dMdt[N_inlet] = dMdt[N_inlet] + 18.57 * Flow_Network[N_inlet, N_inlet] / Volume[N_inlet]

        dIdt[Side_inlet] = dIdt[Side_inlet] + 1 * (0.008 * 0.008 * 3.141592 * 0.1 * 1000) / Volume[Side_inlet]

        dMdt[Side_inlet] = dMdt[Side_inlet] - M[Side_inlet] * (0.008 * 0.008 * 3.141592 * 0.1 * 1000) / Volume[
            Side_inlet]
        dMu_0dt[Side_inlet] = dMu_0dt[Side_inlet] - Mu_0[Side_inlet] * (0.008 * 0.008 * 3.141592 * 0.1 * 1000) / Volume[
            Side_inlet]
        dMu_1dt[Side_inlet] = dMu_1dt[Side_inlet] - Mu_1[Side_inlet] * (0.008 * 0.008 * 3.141592 * 0.1 * 1000) / Volume[
            Side_inlet]
        dMu_2dt[Side_inlet] = dMu_2dt[Side_inlet] - Mu_2[Side_inlet] * (0.008 * 0.008 * 3.141592 * 0.1 * 1000) / Volume[
            Side_inlet]

        L0 = abs((kd * I * f / (ktc + ktd))) ** 0.5

        return np.hstack((dIdt, dMdt, dCTAdt, dMu_0dt, dMu_1dt, dMu_2dt))

    sol = solve_ivp(lambda t, C: fun2(t, C, Flow_Network, Volume), t_span, Initial)
    sol = np.clip(sol.y.T, 0, None)  # [mol/L]

    I = sol[-1, 0 * N: 1 * N]  # 100 - Compartment Initiator Concentration
    M = sol[-1, 1 * N:2 * N]
    CTA = sol[-1, 2 * N:3 * N]
    Mu_0 = sol[-1, 3 * N:4 * N]
    Mu_1 = sol[-1, 4 * N:5 * N]
    Mu_2 = sol[-1, 5 * N:6 * N]

    df = pd.DataFrame({
        'I': I,
        'M': M,
        'CTA': CTA,
        'Mu_0': Mu_0,
        'Mu_1': Mu_1,
        'Mu_2': Mu_2,
    })

    df.to_csv("kinetics_results.csv", index=False)

if __name__ == "__main__":
    os.makedirs('outputs/reaction', exist_ok=True)

    scr_network = 'outputs/flow/flow_network.0.pth'
    dst_network = 'outputs/reaction/flow_network.0.pth'

    scr_checkpoint = 'outputs/flow/optim_checkpoint.0.pth'
    dst_checkpoint = 'outputs/reaction/optim_checkpoint.0.pth'

    shutil.copyfile(scr_network, dst_network)
    shutil.copyfile(scr_checkpoint, dst_checkpoint)

    compartmentalization() # Compartment Formulation
    polymerization() # Reaction Kinetics

    print("Saved Polymerization Reaction Results...")