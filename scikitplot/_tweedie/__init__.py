r"""
scikitplot._tweedie: Tweedie Distribution Module
===============================================

This module implements the Tweedie distribution, a member of the exponential dispersion model (EDM) family, using SciPy's `rv_continuous` class. It is especially useful for modeling claim amounts in the insurance industry, where data often exhibit a mixture of zeroes and positive continuous values.

The primary focus of this package is the compound-Poisson behavior of the Tweedie distribution, particularly in the range `1 < p < 2`. However, it supports calculations for all valid values of the shape parameter `p`.

Classes
-------
tweedie_gen :
    A generator class for Tweedie continuous random variables. Provides parameterization options for users to work with the Tweedie distribution in diverse modeling scenarios.

tweedie :
    An instance of `tweedie_gen`, offering predefined Tweedie distribution functionality.

Features
--------
- Supports modeling data with a point mass at zero and a continuous positive domain.
- Parameterized by a mean and a variance function of the form `Var(Y) = \phi \mu^p`, where `p` is a shape parameter.
- Encompasses well-known distributions as special cases for specific values of `p`.
- Implements SciPy's `rv_continuous` class for seamless integration with Python scientific libraries.

Special Cases of the Tweedie Distribution
-----------------------------------------
The Tweedie distribution family includes several well-known distributions based on the value of the shape parameter `p`:

- `p = 0` : Normal distribution
- `p = 1` : Poisson distribution
- `1 < p < 2` : Compound Poisson-Gamma distribution
- `p = 2` : Gamma distribution
- `2 < p < 3` : Positive stable distributions
- `p = 3` : Inverse Gaussian distribution
- `p > 3` : Positive stable distributions

The Tweedie distribution is undefined for values of `p` in the range `(0, 1)`.

Example Usage
-------------
```python
from scikitplot._tweedie import tweedie

# Generate random variables from a Tweedie distribution
random_data = tweedie.rvs(mu=10, phi=1, p=1.5, size=1000)

# Calculate the probability density function (PDF)
pdf_values = tweedie.pdf(x=random_data, mu=10, phi=1, p=1.5)

# Perform parameter estimation
estimated_params = tweedie.fit(data=random_data)
```

Parameters
----------
mu : float
    The mean of the Tweedie distribution.
phi : float
    The dispersion parameter of the Tweedie distribution.
p : float
    The shape parameter of the Tweedie distribution, which defines its specific form.

Applications
------------
The Tweedie distribution is widely used in:

- Insurance industry: Modeling claim amounts and policy exposure.
- Medical and genomic testing: Analyzing datasets with zero-inflated and continuous positive values.
- Environmental science: Rainfall modeling and hydrology studies.

Notes
-----
The probability density function (PDF) of the Tweedie distribution cannot be expressed in a closed form for most values of `p`. However, approximations and numerical methods are employed to compute the PDF for practical purposes.

References
----------
1. Jørgensen, B. (1987). "Exponential dispersion models". Journal of the Royal Statistical Society, Series B. 49 (2): 127–162.
2. Tweedie, M. C. K. (1984). "An index which distinguishes between some important exponential families". In Statistics: Applications and New Directions. Proceedings of the Indian Statistical Institute Golden Jubilee International Conference.
3. Statistical Methods Series: Zero-Inflated GLM and GLMM. [YouTube]
4https://www.statisticshowto.com/tweedie-distribution/

.. seealso::

   [1] https://www.statsmodels.org/dev/generated/statsmodels.genmod.families.family.Tweedie.html
   [2] https://glum.readthedocs.io/en/latest/glm.html#glum.TweedieDistribution
   [3] https://glum.readthedocs.io/en/latest/glm.html#glum.TweedieDistribution.log_likelihood
"""
# scikitplot/_tweedie/__init__.py

from ._tweedie_dist import *

# Define the tweedie version
# https://pypi.org/project/tweedie/#history
__version__ = "0.0.9"
__author__ = "Peter Quackenbush"
__author_email__ = "pquack@gmail.com"

# Define the tweedie git hash
# scikitplot._build_utils.gitversion.git_remote_version(url='https://github.com/thequackdaddy/tweedie')[0]
__githash__ = 'f14a189d7cd80d41886041f44f40ae4db27d0067'

# Define __all__ to control what gets imported with 'from module import *'
# Combine global names (explicitly defined in the module) and dynamically available names
__all__ = [
  'tweedie_gen',
  'tweedie',
]