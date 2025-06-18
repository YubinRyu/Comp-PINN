from sympy import Symbol, pi, sin, cos, sqrt, Min, Max, Abs, atan2

from modulus.geometry.geometry import Geometry, csg_curve_naming
from modulus.geometry.helper import _sympy_sdf_to_sdf
from modulus.geometry.curve import SympyCurve
from modulus.geometry.parameterization import Parameterization, Parameter, Bounds


class Custom_Plane(Geometry):
    """
    3D Plane perpendicular to z-axis

    Parameters
    ----------
    point_1 : tuple with 3 ints or floats
        lower bound point of plane
    point_2 : tuple with 3 ints or floats
        upper bound point of plane
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, normal=1, parameterization=Parameterization()):
        assert (
                point_1[2] == point_2[2]
        ), "Points must have same coordinate on normal dim"

        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        s_1, s_2 = Symbol(csg_curve_naming(1)), Symbol(csg_curve_naming(2))
        center = (
            point_1[0] + (point_2[0] - point_1[0]) / 2,
            point_1[1] + (point_2[1] - point_1[1]) / 2,
            point_1[2] + (point_2[2] - point_1[2]) / 2,
        )
        side_x = point_2[0] - point_1[0]
        side_y = point_2[1] - point_1[1]

        # surface of the plane
        curve_parameterization = Parameterization({s_1: (-1, 1), s_2: (-1, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] + 0.5 * s_2 * side_y,
                "z": center[2],
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1e-10 + normal,  # TODO rm 1e-10
            },
            parameterization=curve_parameterization,
            area=side_x * side_y,
        )
        curves = [curve_1]

        # calculate SDF
        sdf = normal * (center[2] - z)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
                Parameter("z"): (point_1[2], point_2[2]),
            },
            parameterization=parameterization,
        )

        # initialize Plane
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class Box(Geometry):
    """
    3D Box/Cuboid

    Parameters
    ----------
    point_1 : tuple with 3 ints or floats
        lower bound point of box
    point_2 : tuple with 3 ints or floats
        upper bound point of box
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        s_1, s_2 = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        center = (
            point_1[0] + (point_2[0] - point_1[0]) / 2,
            point_1[1] + (point_2[1] - point_1[1]) / 2,
            point_1[2] + (point_2[2] - point_1[2]) / 2,
        )
        side_x = point_2[0] - point_1[0]
        side_y = point_2[1] - point_1[1]
        side_z = point_2[2] - point_1[2]

        # surface of the box
        curve_parameterization = Parameterization({s_1: (-1, 1), s_2: (-1, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] + 0.5 * s_2 * side_y,
                "z": center[2] + 0.5 * side_z,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1,
            },
            parameterization=curve_parameterization,
            area=side_x * side_y,
        )
        curve_2 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] + 0.5 * s_2 * side_y,
                "z": center[2] - 0.5 * side_z,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=side_x * side_y,
        )
        curve_3 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] + 0.5 * side_y,
                "z": center[2] + 0.5 * s_2 * side_z,
                "normal_x": 0,
                "normal_y": 1,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side_x * side_z,
        )
        curve_4 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * s_1 * side_x,
                "y": center[1] - 0.5 * side_y,
                "z": center[2] + 0.5 * s_2 * side_z,
                "normal_x": 0,
                "normal_y": -1,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side_x * side_z,
        )
        curve_5 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * side_x,
                "y": center[1] + 0.5 * s_1 * side_y,
                "z": center[2] + 0.5 * s_2 * side_z,
                "normal_x": 1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side_y * side_z,
        )
        curve_6 = SympyCurve(
            functions={
                "x": center[0] - 0.5 * side_x,
                "y": center[1] + 0.5 * s_1 * side_y,
                "z": center[2] + 0.5 * s_2 * side_z,
                "normal_x": -1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=side_y * side_z,
        )
        curves = [curve_1, curve_2, curve_3, curve_4, curve_5, curve_6]

        # calculate SDF
        x_dist = Abs(x - center[0]) - 0.5 * side_x
        y_dist = Abs(y - center[1]) - 0.5 * side_y
        z_dist = Abs(z - center[2]) - 0.5 * side_z
        outside_distance = sqrt(
            Max(x_dist, 0) ** 2 + Max(y_dist, 0) ** 2 + Max(z_dist, 0) ** 2
        )
        inside_distance = Min(Max(x_dist, y_dist, z_dist), 0)
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
                Parameter("z"): (point_1[2], point_2[2]),
            },
            parameterization=parameterization,
        )

        # initialize Box
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class custom_Cylinder(Geometry):
    """
3D Infinite Cylinder
Axis parallel to z-axis, no caps on ends

Parameters
----------
center : tuple with 3 ints or floats
center of cylinder
radius : int or float
radius of cylinder
height : int or float
height of cylinder
parameterization : Parameterization
Parameterization of geometry.
"""

    def __init__(self, center, radius, height, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        h, r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        theta = Symbol(csg_curve_naming(2))

        # surface of the cylinder
        curve_parameterization = Parameterization(
            {h: (-1, 1), r: (0, 1), theta: (0, 2 * pi)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + radius * cos(theta),
                "y": center[1] + radius * sin(theta),
                "z": center[2] + 0.5 * h * height,
                "normal_x": 1 * cos(theta),
                "normal_y": 1 * sin(theta),
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=height * 2 * pi * radius,
        )
        curves = [curve_1]

        # calculate SDF
        r_dist = sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        sdf2 = z - (center[2] - 0.5 * height)

        sdf3 = radius - r_dist

        sdf = Min(sdf3, sdf2)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - radius, center[0] + radius),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - height / 2, center[2] + height / 2),
            },
            parameterization=parameterization,
        )

        # initialize Infinite Cylinder
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class Cylinder(Geometry):
    """
    3D Cylinder
    Axis parallel to z-axis

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of cylinder
    radius : int or float
        radius of cylinder
    height : int or float
        height of cylinder
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, height, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        h, r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        theta = Symbol(csg_curve_naming(2))

        # surface of the cylinder
        curve_parameterization = Parameterization(
            {h: (-1, 1), r: (0, 1), theta: (0, 2 * pi)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + radius * cos(theta),
                "y": center[1] + radius * sin(theta),
                "z": center[2] + 0.5 * h * height,
                "normal_x": 1 * cos(theta),
                "normal_y": 1 * sin(theta),
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=height * 2 * pi * radius,
        )
        curve_2 = SympyCurve(
            functions={
                "x": center[0] + sqrt(r) * radius * cos(theta),
                "y": center[1] + sqrt(r) * radius * sin(theta),
                "z": center[2] + 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": 1,
            },
            parameterization=curve_parameterization,
            area=pi * radius ** 2,
        )
        curve_3 = SympyCurve(
            functions={
                "x": center[0] + sqrt(r) * radius * cos(theta),
                "y": center[1] + sqrt(r) * radius * sin(theta),
                "z": center[2] - 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=pi * radius ** 2,
        )
        curves = [curve_1, curve_2, curve_3]

        # calculate SDF
        r_dist = sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        z_dist = Abs(z - center[2])
        outside_distance = sqrt(
            Min(0, radius - r_dist) ** 2 + Min(0, 0.5 * height - z_dist) ** 2
        )
        inside_distance = -1 * Min(
            Abs(Min(0, r_dist - radius)), Abs(Min(0, z_dist - 0.5 * height))
        )
        sdf = -(outside_distance + inside_distance)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - radius, center[0] + radius),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - height / 2, center[2] + height / 2),
            },
            parameterization=parameterization,
        )

        # initialize Cylinder
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )

class Tangential_flow(Geometry):
    """
    Vertical plane passing through the central axis of a cylinder at a given theta
    with limited z and r ranges, and constrained to the quarter containing the angle theta.

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of the cylinder base
    radius : int or float
        radius of the cylinder
    height : int or float
        height of the cylinder
    theta : float
        angle in radians where the vertical plane crosses the cylinder
    z_min : float
        minimum value of z (height)
    z_max : float
        maximum value of z (height)
    r_min : float
        minimum value of r (radial distance from center)
    r_max : float
        maximum value of r (radial distance from center)
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, height, theta, z_min, z_max, r_min, r_max, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        h = Symbol(csg_curve_naming(0))
        r = Symbol(csg_curve_naming(1))

        # Normalize theta to be within [0, 2*pi]
        theta = theta % (2 * pi)

        # Normal vector components of the vertical plane
        normal_x = -sin(theta)
        normal_y = cos(theta)
        normal_z = 0  # For a vertical plane, the normal in the z-direction is zero

        # Equation of the vertical plane passing through the central axis
        plane_equation = normal_x * (x - center[0]) + normal_y * (y - center[1])

        # Define the SDF for the vertical plane
        sdf_plane = plane_equation

        # Define the SDF for the radial boundary
        r_dist = sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        sdf_r_min = r_min - r_dist  # Outside distance for r_min
        sdf_r_max = r_dist - r_max  # Inside distance for r_max
        sdf_r = Max(sdf_r_min, sdf_r_max)  # Combine r_min and r_max SDFs

        # Define the SDF for the z boundary
        sdf_z_min = z_min - z  # Below z_min
        sdf_z_max = z - z_max  # Above z_max
        sdf_z = Max(sdf_z_min, sdf_z_max)  # Combine z_min and z_max SDFs

        # Additional SDF to restrict the plane within the specified quarter
        # Calculate the angle of each point (x, y) in the x-y plane relative to the center
        point_theta = atan2(y - center[1], x - center[0])
        point_theta = point_theta % (2 * pi)  # Normalize to [0, 2*pi]

        # Calculate the min and max angles for the quarter containing theta
        theta_min = (theta - pi / 4) % (2 * pi)
        theta_max = (theta + pi / 4) % (2 * pi)

        # Define SDF for the angular constraint
        if theta_min < theta_max:
            # Standard case
            sdf_theta_restriction = Max(theta_min - point_theta, point_theta - theta_max)
        else:
            # Wrap-around case
            sdf_theta_restriction = -Min(theta_min - point_theta, point_theta - theta_max)

        # Combine the SDFs for plane, r range, z range, and angular constraint
        sdf = Max(sdf_plane, sdf_r, sdf_z, sdf_theta_restriction)

        # Define a line along z-axis for visualization purposes, respecting r and z limits
        curve_parameterization = Parameterization(
            {h: (-1, 1), r: (r_min, r_max)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )

        # A line along z-axis in the x-y plane at the specified angle, with limited r and z
        curve = SympyCurve(
            functions={
                "x": center[0] + r * cos(theta),  # Starting point of the line
                "y": center[1] + r * sin(theta),
                "z": center[2] + h * (z_max - z_min) / 2,  # Line segment with height limit
                "normal_x": normal_x,
                "normal_y": normal_y,
                "normal_z": normal_z,
            },
            parameterization=curve_parameterization,
            area=(r_max - r_min) * (z_max - z_min)  # Area based on r and z limits
        )
        curves = [curve]

        # Calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - radius, center[0] + radius),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (z_min, z_max),
            },
            parameterization=parameterization,
        )

        # Initialize Vertical Plane geometry
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class Axial_flow(Geometry):
    """
    Circular plane at the bottom of a cylinder with limited theta range.
    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of the circular plane
    radius : int or float
        radius of the circular plane
    height : int or float
        height of the cylinder (used for z location)
    theta_min : float
        minimum value of theta (in radians)
    theta_max : float
        maximum value of theta (in radians)
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, height, theta_min, theta_max, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        h, r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        theta = Symbol(csg_curve_naming(2))

        # Restrict theta range to (theta_min, theta_max)
        curve_parameterization = Parameterization(
            {h: (-1, 1), r: (0, 1), theta: (theta_min, theta_max)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )

        # Define the circular plane at the bottom with limited theta range
        curve_3 = SympyCurve(
            functions={
                "x": center[0] + sqrt(r) * radius * cos(theta),
                "y": center[1] + sqrt(r) * radius * sin(theta),
                "z": center[2] - 0.5 * height,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=(theta_max - theta_min) * radius ** 2 / 2,  # Adjust area based on limited theta
        )
        curves = [curve_3]

        # Calculate SDF
        r_dist = sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        z_dist = Abs(z - center[2])
        outside_distance = sqrt(
            Min(0, radius - r_dist) ** 2 + Min(0, 0.5 * height - z_dist) ** 2
        )
        inside_distance = -1 * Min(
            Abs(Min(0, r_dist - radius)), Abs(Min(0, z_dist - 0.5 * height))
        )
        sdf = -(outside_distance + inside_distance)

        # Calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - radius, center[0] + radius),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - height / 2, center[2] + height / 2),
            },
            parameterization=parameterization,
        )

        # Initialize Cylindrical_plane
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class Radial_flow(Geometry):
    """
    3D Cylinder with limited theta range
    Axis parallel to z-axis
    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of cylinder
    radius : int or float
        radius of cylinder
    height : int or float
        height of cylinder
    theta_min : float
        minimum value of theta (in radians)
    theta_max : float
        maximum value of theta (in radians)
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, height, theta_min, theta_max, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        h, r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        theta = Symbol(csg_curve_naming(2))

        # Restrict the theta range
        curve_parameterization = Parameterization(
            {h: (-1, 1), r: (0, 1), theta: (theta_min, theta_max)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )

        # Define the surface of the cylinder with limited theta
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + radius * cos(theta),
                "y": center[1] + radius * sin(theta),
                "z": center[2] + 0.5 * h * height,
                "normal_x": -1 * cos(theta),
                "normal_y": -1 * sin(theta),
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=height * (theta_max - theta_min) * radius,  # Adjusted for theta range
        )
        curves = [curve_1]

        # Calculate SDF
        r_dist = sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        z_dist = Abs(z - center[2])
        outside_distance = sqrt(
            Min(0, radius - r_dist) ** 2 + Min(0, 0.5 * height - z_dist) ** 2
        )
        inside_distance = -1 * Min(
            Abs(Min(0, r_dist - radius)), Abs(Min(0, z_dist - 0.5 * height))
        )
        sdf = -(outside_distance + inside_distance)

        # Calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - radius, center[0] + radius),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - height / 2, center[2] + height / 2),
            },
            parameterization=parameterization,
        )

        # Initialize Cylinder
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )