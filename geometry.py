import numpy as np
from custom_primitives_3d import *
from modulus.geometry import Parameterization

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
z_pos = Symbol("z_pos")
r = Symbol("r")
h = Symbol("h")
str = Symbol("str")

r_range = (-1.0, 1.0)
h_range = (-1.0, 1.0)
str_range = (-1.0, 1.0)

param_range = {
    r: r_range,
    h: h_range,
    str: str_range,
}

radius = 0.05 + 0.02 * r
height = 0.045 * h
stirring_rate = 10.0 * str

class Reactor(object):
    def __init__(self, compartment: bool = False):
        # set param range
        pr = Parameterization(param_range)
        self.pr = param_range

        cylinder_center = (0.0, 0.0, 0.0)
        cylinder_radius = 0.1
        cylinder_height = 0.2

        impeller_center = (0.0, 0.0, 0.035)
        impeller_radius = 0.0025
        impeller_height = 0.135

        blade_1_origin1 = (-radius, -0.005, height - 0.015)
        blade_1_origin2 = (radius, 0.005, height + 0.015)
        
        blade_2_origin1 = (-0.005, -radius, height - 0.015)
        blade_2_origin2 = (0.005, radius, height + 0.015)

        inlet_origin = (-0.1, -0.1, 0.1)
        inlet_dim = (0.2, 0.2, 0.0)

        outlet_origin = (-0.1, -0.1, -0.1)
        outlet_dim = (0.2, 0.2, 0.0)

        plane_origin = (-0.1, -0.1)
        plane_dim = (0.2, 0.2)

        # Tank
        self.cylinder = custom_Cylinder(
            cylinder_center,
            cylinder_radius,
            cylinder_height,
            parameterization=pr,
        )
        self.cylinder_total = self.cylinder

        # Impeller
        self.impeller = Cylinder(
            impeller_center,
            impeller_radius,
            impeller_height,
            parameterization=pr,
        )

        self.blade_1 = Box(
            blade_1_origin1,
            blade_1_origin2,
            parameterization=pr,
        )

        self.blade_2 = Box(
            blade_2_origin1,
            blade_2_origin2,
            parameterization=pr,
        )

        self.inlet_plane = Custom_Plane(
            (inlet_origin[0], inlet_origin[1], inlet_origin[2]),
            (
                inlet_origin[0] + inlet_dim[0],
                inlet_origin[1] + inlet_dim[1],
                inlet_origin[2]
            ),
            -1,
            parameterization=pr,
        )

        # planes for outlet
        self.outlet_plane = Custom_Plane(
            (outlet_origin[0], outlet_origin[1], outlet_origin[2]),
            (
                outlet_origin[0] + outlet_dim[0],
                outlet_origin[1] + outlet_dim[1],
                outlet_origin[2]
            ),
            1,
            parameterization=pr,
        )

        # planes for integral continuity
        self.integral_plane = Custom_Plane(
            (plane_origin[0], plane_origin[1], z_pos),
            (plane_origin[0] + plane_dim[0], plane_origin[1] + plane_dim[1], z_pos),
            -1,
            parameterization=pr,
        )

        self.impeller_total = self.blade_1 + self.blade_2 + self.impeller
        self.geo = self.cylinder_total - self.impeller_total

        if compartment == True:
            z_list = list([0.04, 0.04, 0.04, 0.04])
            r_list = list([0.03, 0.025, 0.015, 0.015])
            t_cor = list([47, 135, 225, 313])

            t_cor = np.array(t_cor) * np.pi / 180
            t_list = [t_cor[0] - t_cor[-1] + 2 * np.pi] + [t_cor[x + 1] - t_cor[x] for x in range(len(t_cor) - 1)]

            self.t_list = t_list
            self.t_cor = t_cor
            self.T_len = len(t_list)
            T_len = self.T_len

            z_cor = [0.1] + [0.1 - sum(z_list[:x + 1]) for x in range(len(z_list))] + [-0.1]
            self.z_list = z_list
            self.z_cor = z_cor
            z_list = z_list + [z_cor[-2] - z_cor[-1]]
            self.Z_len = len(z_list)
            Z_len = self.Z_len

            r_cor = [0.0000001] + [sum(r_list[:x + 1]) for x in range(len(r_list))] + [0.1]
            self.r_list = r_list
            self.r_cor = r_cor
            r_list = r_list + [r_cor[-1] - r_cor[-2]]
            self.R_len = len(r_list)
            R_len = self.R_len

            Compart_N = R_len * Z_len * T_len
            self.Flow_network = []

            for i in range(Compart_N):
                Test_list = []
                for j in range(Compart_N):
                    Test_list.append(0)
                self.Flow_network.append(Test_list)

            self.Cor = []
            self.Cor1 = []
            delta_theta = (t_cor[0] - t_cor[0 - 1]) % (2 * np.pi)
            self.Flow_network2 = Radial_flow(
                (cylinder_center[0], cylinder_center[1], z_cor[0] - z_list[0] / 2),
                r_cor[1],
                z_list[0], t_cor[0] - delta_theta, t_cor[0]
            )

            for t in range(len(t_cor)):
                for i in range(len(z_list)):
                    for j in range(len(r_list) - 1):
                        delta_theta = (t_cor[t] - t_cor[t - 1]) % (2 * np.pi)

                        self.Flow_network[t * R_len * Z_len + i * R_len + j][
                            t * R_len * Z_len + i * R_len + j + 1] = Radial_flow(
                            (cylinder_center[0], cylinder_center[1], z_cor[i] - z_list[i] / 2),
                            r_cor[j + 1],
                            z_list[i], t_cor[t] - delta_theta, t_cor[t],
                            parameterization=pr,
                        )

                        self.Cor.append([t * R_len * Z_len + i * R_len + j, t * R_len * Z_len + i * R_len + j + 1])
                        self.Cor1.append([t * R_len * Z_len + i * R_len + j, t * R_len * Z_len + i * R_len + j + 1])

            self.Cor2 = []
            self.Cor3 = []

            for t in range(len(t_cor)):
                for i in range(len(z_list) - 1):
                    for j in range(len(r_list)):
                        delta_theta = (t_cor[t] - t_cor[t - 1]) % (2 * np.pi)

                        self.Bigger = Axial_flow(
                            (cylinder_center[0], cylinder_center[1], z_cor[i] - z_list[i] / 2),
                            r_cor[j + 1],
                            z_list[i], t_cor[t] - delta_theta, t_cor[t],
                            parameterization=pr,
                        )

                        self.Smaller = Axial_flow(
                            (cylinder_center[0], cylinder_center[1], z_cor[i] - z_list[i] / 2),
                            r_cor[j],
                            z_list[i], t_cor[t] - delta_theta, t_cor[t],
                            parameterization=pr,
                        )

                        self.Flow_network[t * R_len * Z_len + i * R_len + j][
                            t * R_len * Z_len + i * R_len + j + R_len] = self.Bigger - self.Smaller
                        self.Cor2.append([t * R_len * Z_len + i * R_len + j, t * R_len * Z_len + i * R_len + j + R_len])
                        self.Cor.append([t * R_len * Z_len + i * R_len + j, t * R_len * Z_len + i * R_len + j + R_len])

            self.Cos = []
            for t in range(len(t_cor)):
                for i in range(len(z_list)):
                    for j in range(len(r_list)):
                        self.tan = Tangential_flow(
                            (cylinder_center[0], cylinder_center[1], z_cor[i] - z_list[i] / 2),
                            0.1,  # r_cor[j+1],
                            0.2, t_cor[t], z_cor[i + 1], z_cor[i], r_cor[j], r_cor[j + 1],
                            parameterization=pr,
                        )
                        t_1 = (t + 1) % T_len

                        self.Flow_network[t * R_len * Z_len + i * R_len + j][
                            (t_1) * R_len * Z_len + i * R_len + j] = self.tan
                        self.Cor3.append([t * R_len * Z_len + i * R_len + j, (t_1) * R_len * Z_len + i * R_len + j])
                        self.Cor.append([t * R_len * Z_len + i * R_len + j, (t_1) * R_len * Z_len + i * R_len + j])
                        self.Cos.append(np.cos(t_cor[t]))

            Network_boolean = np.zeros((Compart_N, Compart_N))
            self.Net = Network_boolean

            for i in self.Cor:
                Network_boolean[i[0], i[1]] = 1

            self.volume = []
            for t in range(len(t_cor)):
                for i in range(len(z_list)):
                    for j in range(len(r_list)):
                        delta_theta = (t_cor[t] - t_cor[t - 1]) % (2 * np.pi)
                        self.volume.append(
                            ((r_cor[j + 1] * r_cor[j + 1] - r_cor[j] * r_cor[j]) * 3.141592 * z_list[i]) * (
                                delta_theta) / (2 * np.pi))