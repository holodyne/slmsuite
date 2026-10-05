"""
Common definitions.
"""

import numpy as np

INTEGER_TYPES = (
    int,
    np.integer,
)

FLOAT_TYPES = (
    float,
    np.floating,
)

REAL_TYPES = (
    *INTEGER_TYPES,
    *FLOAT_TYPES,
)

SCALAR_TYPES = (*REAL_TYPES, complex, np.complexfloating)


def iseven(x):
    """
    Test if integer(s) are even.

    Parameters
    ----------
    x : int OR list of int
        The integer(s) to test.

    Returns
    -------
    bool OR list of bool
        Whether or not ``x`` is even.
    """
    return (np.around(x).astype(int) & 0x1) == 0
