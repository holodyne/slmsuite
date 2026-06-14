"""
Repository of common analytic phase patterns.
"""
from slmsuite.holography.toolbox.phase._lenses import *
from slmsuite.holography.toolbox.phase._gratings import *
from slmsuite.holography.toolbox.phase._zernike import *
from slmsuite.holography.toolbox.phase._structured import *
from slmsuite.holography.toolbox.phase._misc import *

# Re-export private names needed by other modules (not picked up by `import *`)
from slmsuite.holography.toolbox.phase._lenses import _parse_focal_length
from slmsuite.holography.toolbox.phase._misc import _determine_source_radius
from slmsuite.holography.toolbox.phase._structured import _ince_polynomial
from slmsuite.holography.toolbox.phase._zernike import (
    CUDA_KERNELS,
    _load_cuda,
    _zernike_indices_parse,
    _zernike_build_order,
    _zernike_build_indices,
    _zernike_coefficients,
    _zernike_populate_basis_map,
    _cantor_pairing,
    _inverse_cantor_pairing,
    _parse_out,
)