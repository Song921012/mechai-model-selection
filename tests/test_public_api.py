import math
import unittest
import warnings

import torch

from mechai_model_selection import (
    ObservableGeometry,
    effective_dimension,
    gic_laplace,
    observable_dimension,
    relative_volume,
)


class PublicApiTests(unittest.TestCase):
    def test_canonical_geometry_names(self):
        values = torch.tensor([9.0, 1.0, 0.0], dtype=torch.float64)
        self.assertAlmostEqual(float(effective_dimension(values)), 1.4)
        self.assertAlmostEqual(float(relative_volume(values)), math.log(20.0))

    def test_laplace_name_and_compatibility_warning(self):
        geometry = ObservableGeometry(torch.tensor([9.0, 1.0], dtype=torch.float64))
        self.assertAlmostEqual(gic_laplace(10.0, geometry, prior_energy=2.0), 12.0 + math.log(20.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            observable_dimension(geometry.eigenvalues)
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))


if __name__ == "__main__":
    unittest.main()
