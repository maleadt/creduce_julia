Rmath.jl
========

[![Rmath](http://pkg.julialang.org/badges/Rmath_0.4.svg)](http://pkg.julialang.org/?pkg=Rmath)
[![Rmath](http://pkg.julialang.org/badges/Rmath_0.5.svg)](http://pkg.julialang.org/?pkg=Rmath)
[![Travis](https://travis-ci.org/JuliaStats/Rmath.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/Rmath.jl)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/github/JuliaStats/Rmath.jl?svg=true&branch=master)](https://ci.appveyor.com/project/andreasnoack/rmath-jl/branch/master)

Archive of functions that emulate R's d-p-q-r functions for probability distributions.

This package will download [libRmath-julia](https://github.com/staticfloat/RmathBuilder)
on Julia 0.6 and newer where it is no longer built as part of Julia. Note that
libRmath-julia is licensed under the GPLv2, see https://github.com/JuliaLang/Rmath-julia/blob/master/COPYING.
The Julia bindings here are licensed under [MIT](LICENSE.md).

If you only want to use this package for the sake of building the Rmath library, you can do
```julia
Pkg.add("Rmath")
Pkg.build("Rmath")
import Rmath: libRmath
libRmath
```
