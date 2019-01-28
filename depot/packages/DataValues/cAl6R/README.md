# DataValues

[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Build Status](https://travis-ci.org/queryverse/DataValues.jl.svg?branch=master)](https://travis-ci.org/queryverse/DataValues.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/nkd83orhu4xm68yp/branch/master?svg=true)](https://ci.appveyor.com/project/queryverse/datavalues-jl/branch/master)
[![DataValues](http://pkg.julialang.org/badges/DataValues_0.6.svg)](http://pkg.julialang.org/?pkg=DataValues)
[![codecov](https://codecov.io/gh/queryverse/DataValues.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/queryverse/DataValues.jl)

## Overview

This package provides the type ``DataValue`` that is used to represent missing data.

Currently the main use of this type is in the [Query.jl](https://github.com/queryverse/Query.jl) and [IterableTables.jl](https://github.com/queryverse/IterableTables.jl) packages.

This repo is based on the following principles/ideas:

- This type is meant to make life for data scientists as easy as possible.
That is the main guiding principle.
- We hook into the dot broadcasting mechanism from julia 0.7 to provide
lifting functionality for functions that don't have specific methods
for ``DataValue`` arguments.
- The ``&`` and ``|`` operators follow the [3VL](https://en.wikipedia.org/wiki/Three-valued_logic)
semantics for ``DataValue``s.
- Comparison operators like ``==``, ``<`` etc. on ``DataValue``s return
``Bool`` values, i.e. they are normal [predicates](https://en.wikipedia.org/wiki/Predicate_(mathematical_logic)).
- The package provides many lifted methods.
- One can access or unpack the value within a ``DataValue`` either via the
``get(x)`` function, or use the ``x[]`` syntax.

Any help with this package would be greatly appreciated!
