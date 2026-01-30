import numpy as np
from numpy.typing import NDArray


__all__ = (
    "hwp",
    "qwp",
    "waveplate",
    "rotator",
    "generic",
    "linear_polarizer",
    "wollaston",
)


def hwp(theta=0) -> NDArray:
    """Return the Mueller matrix for an ideal half-wave plate (HWP) with fast-axis oriented to the given angle, in radians.

    Parameters
    ----------
    theta : float, optional
        Rotation angle of the fast-axis, in radians. By default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix representing the HWP

    Examples
    --------
    >>> hwp(0)
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0., -1.,  0.],
           [ 0.,  0.,  0., -1.]])

    >>> hwp(np.deg2rad(45))
    array([[ 1.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0., -1.]])


    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    M = np.array(
        (
            (1, 0, 0, 0),
            (0, cos2t**2 - sin2t**2, 2 * cos2t * sin2t, 0),
            (0, 2 * cos2t * sin2t, sin2t**2 - cos2t**2, 0),
            (0, 0, 0, -1),
        )
    )
    return M


def qwp(theta=0) -> NDArray:
    """Return the Mueller matrix for an ideal quarter-wave plate (QWP) with fast-axis oriented to the given angle, in radians.

    Parameters
    ----------
    theta : float, optional
        Rotation angle of the fast-axis in radians. By default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix representing the QWP

    Examples
    --------
    >>> qwp(0)
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -0.],
           [ 0.,  0.,  0.,  1.],
           [ 0.,  0., -1.,  0.]])

    >>> qwp(np.deg2rad(45))
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0., -1.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  1., -0.,  0.]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    M = np.array(
        (
            (1, 0, 0, 0),
            (0, cos2t**2, cos2t * sin2t, -sin2t),
            (0, cos2t * sin2t, sin2t**2, cos2t),
            (0, sin2t, -cos2t, 0),
        )
    )
    return M


def waveplate(theta=0, delta=0) -> NDArray:
    """Return the Mueller matrix for a waveplate with arbitrary phase retardance.

    Parameters
    ----------
    theta : float, optional
        Rotation angle of the fast-axis in radians. By default 0
    delta : float, optional
        Retardance in radians, by default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix representing the waveplate

    Examples
    --------
    >>> waveplate(0, np.pi) # HWP
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -0.],
           [ 0.,  0., -1.,  0.],
           [ 0.,  0., -0., -1.]])

    >>> waveplate(np.deg2rad(45), np.pi/2) # QWP at 45°
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0., -1.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  1., -0.,  0.]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    cosd = np.cos(delta)
    sind = np.sin(delta)
    a = (1 - cosd) * sin2t * cos2t
    M = np.array(
        (
            (1, 0, 0, 0),
            (0, cos2t**2 + cosd * sin2t**2, a, -sind * sin2t),
            (0, a, sin2t**2 + cosd * cos2t**2, sind * cos2t),
            (0, sind * sin2t, -sind * cos2t, cosd),
        )
    )
    return M


def generic(theta=0, epsilon=0, delta=0) -> NDArray:
    """Return a generic optic with diattenuation ``epsilon`` and phase retardance ``delta`` oriented at angle ``theta``.

    Parameters
    ----------
    theta : float, optional
        Rotation angle of the fast-axis in radians, by default 0
    epsilon : float, optional
        Diattenuation, by default 0
    delta : float, optional
        Retardance in radians, by default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix for the optic

    Examples
    --------
    >>> generic() # Identity
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0., -0.,  1.]])

    >>> generic(epsilon=0.01, delta=np.pi) # mirror with diatt.
    array([[ 1.     ,  0.01   ,  0.     ,  0.     ],
           [ 0.01   ,  1.     ,  0.     , -0.     ],
           [ 0.     ,  0.     , -0.99995,  0.     ],
           [ 0.     ,  0.     , -0.     , -0.99995]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    cosd = np.cos(delta)
    sind = np.sin(delta)
    fac = np.sqrt((1 - epsilon) * (1 + epsilon))
    M = np.array(
        (
            (1, epsilon * cos2t, epsilon * sin2t, 0),
            (
                epsilon * cos2t,
                cos2t**2 + sin2t**2 * fac * cosd,
                cos2t * sin2t - fac * cosd * cos2t * sin2t,
                -fac * sind * sin2t,
            ),
            (
                epsilon * sin2t,
                cos2t * sin2t - fac * cosd * cos2t * sin2t,
                sin2t**2 + cos2t**2 * fac * cosd,
                fac * sind * cos2t,
            ),
            (0, fac * sind * sin2t, -fac * sind * cos2t, fac * cosd),
        )
    )

    return M


def rotator(theta=0) -> NDArray:
    """Return the Mueller matrix for rotation clockwise about the optical axis.

    Parameters
    ----------
    theta : float, optional
        Angle of rotation, in radians. By default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix

    Examples
    --------
    >>> rotator(0)
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0., -0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])

    >>> rotator(np.deg2rad(45))
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0., -1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    M = np.array(
        ((1, 0, 0, 0), (0, cos2t, sin2t, 0), (0, -sin2t, cos2t, 0), (0, 0, 0, 1))
    )
    return M


def linear_polarizer(theta=0) -> NDArray:
    """Return the Mueller matrix for an ideal linear polarizer oriented at the given angle.

    Parameters
    ----------
    theta : float, optional
        Angle of rotation, in radians. By default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix for the linear polarizer

    Examples
    --------
    >>> linear_polarizer(0)
    array([[0.5, 0.5, 0. , 0. ],
           [0.5, 0.5, 0. , 0. ],
           [0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. ]])

    >>> linear_polarizer(np.deg2rad(45))
    array([[0.5, 0. , 0.5, 0. ],
           [0. , 0. , 0. , 0. ],
           [0.5, 0. , 0.5, 0. ],
           [0. , 0. , 0. , 0. ]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    M = np.array(
        (
            (1, cos2t, sin2t, 0),
            (cos2t, cos2t**2, cos2t * sin2t, 0),
            (sin2t, cos2t * sin2t, sin2t**2, 0),
            (0, 0, 0, 0),
        )
    )
    return 0.5 * M


def mirror() -> NDArray:
    """Return the Mueller matrix for an ideal mirror.

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix for the mirror

    Examples
    --------
    >>> mirror()
    array([[ 1,  0,  0,  0],
           [ 0,  1,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0,  0, -1]])
    """
    return hwp(theta=0)


def wollaston(ordinary: bool = True, epsilon=1) -> NDArray:
    """Return the Mueller matrix for a Wollaston prism or polarizing beamsplitter.

    Parameters
    ----------
    ordinary : bool, optional
        Return the ordinary beam's Mueller matrix, by default True
    epsilon : float, optional
        For imperfect beamsplitters, the diattenuation of the optic, by default 1

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix for the selected output beam

    Examples
    --------
    >>> wollaston()
    array([[0.5, 0.5, 0. , 0. ],
           [0.5, 0.5, 0. , 0. ],
           [0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. ]])

    >>> wollaston(False, epsilon=0.8)
    array([[ 0.5, -0.4,  0. ,  0. ],
           [-0.4,  0.5,  0. ,  0. ],
           [ 0. ,  0. ,  0.3,  0. ],
           [ 0. ,  0. ,  0. ,  0.3]])
    """
    epsilon = epsilon if ordinary else -epsilon

    radicand = (1 - epsilon) * (1 + epsilon)
    M = np.array(
        (
            (1, epsilon, 0, 0),
            (epsilon, 1, 0, 0),
            (0, 0, np.sqrt(radicand), 0),
            (0, 0, 0, np.sqrt(radicand)),
        )
    )
    return 0.5 * M


def instrumental(pQ=0, pU=0, pV=0):
    M = np.eye(4)
    M[1:, 0] = (pQ, pU, pV)
    return M


def elliptical_retarder(theta=0, delta_h=0, delta_45=0, delta_r=0):
    # eqn 6.28 Chipman
    delta = np.sqrt(delta_h**2 + delta_45**2 + delta_r**2)
    s = np.sin(delta)
    c = np.cos(delta)
    t = 1 - c
    m11 = (delta_h**2 + (delta_45**2 + delta_r**2) * c) / delta**2
    m12 = delta_45 * delta_h * t / delta**2 + delta_r * s / delta
    m13 = delta_h * delta_r * t / delta**2 - delta_45 * s / delta
    m21 = delta_45 * delta_h * t / delta**2 - delta_r * s / delta
    m22 = (delta_45**2 + (delta_r**2 + delta_h**2) * c) / delta**2
    m23 = delta_r * delta_45 * t / delta**2 + delta_h * s / delta
    m31 = delta_h * delta_r * t / delta**2 + delta_45 * s / delta
    m32 = delta_r * delta_45 * t / delta**2 - delta_h * s / delta
    m33 = (delta_r**2 + (delta_45**2 + delta_h**2) * c) / delta**2
    M = np.array(
        [
            [
                1,
                0,
                0,
                0,
            ],
            [0, m11, m12, m13],
            [0, m21, m22, m23],
            [0, m31, m32, m33],
        ]
    )
    R = rotator(theta)
    return R.T @ M @ R

def elliptical_diattenuater_retarder(theta=0, epsilon=0, delta_h=0, delta_45=0, delta_r=0):
    M_retarder = elliptical_retarder(delta_h=delta_h, delta_45=delta_45, delta_r=delta_r)
    M_diatt = generic(epsilon=epsilon)
    R = rotator(theta)
    return R.T @ M_diatt @ M_retarder @ R
