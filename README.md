# C-Reduce for Julia

This repository contains a set of scripts to reduce Julia test cases with C-Reduce. It can
be helpful if you want to minimize code that causes a crash or bug, without having to
iteratively edit a script and manually verify the bug still occurs.

Note that testcase reduction works best if the testcase can be evaluated quickly (i.e.,
seconds rather than minutes). If your testcase takes a long time to run, you may want to do
some manual editing first to get rid of dependencies that take a long time to load.


## Requirements

- [C-Reduce](https://github.com/csmith-project/creduce) 2.8.0+, on PATH
- Julia 1.0+, on PATH


## Usage

Start by **putting your test case in `main.jl`**. If your script needs any
packages, activate a Julia instance in the C-Reduce environment by executing the
`julia` wrapper script from the repository's root directory:

```
$ cd /path/to/checkout
$ ./julia
] add ...
] dev ...
```

Packages that you `add` will be ignored by the reduction process; only packages you `dev`
will take part in it.

Next, **modify the `run` script** to properly catch the error you are dealing with and
return 0 if the reduced file is good. Often, you want to look for specific output in the
standard error stream. If you need to use any Julia flags, or want to use a specific build,
edit the `julia` wrapper script accordingly. You may also want to edit the `reduce` script
to adjust the timeout, which defaults to 1 minute.

Optionally, **preprocess the source** to get rid of irrelevant source code by running the
`tools/preprocess.jl` script. It is recommended to remove other irrelevant sources as well,
such as tests, examples, or documentation. Always verify that the `run` script still returns
0 afterwards.

Finally, **execute the `reduce` script**. This should finalize the environment and start
C-Reduce.


## Demo

Let's start with a `main.jl` that prints `Hello, world!`:

```julia
println("Hello, World!")
println("1 + 1 = ", 1 + 1)
exit(0)
```

We can verify this executes correctly by running the `julia` wrapper script:

```
❯ ./julia main.jl
Hello, World!
1 + 1 = 2
```

Say we are only interested in the `Hello, World!`, so we adapt the `run` script as follows
(keeping the surrounding boilerplate, which among other things ensures the process exited
successfully):

```
...
$DIR/julia main.jl |& grep "example error message"
...
```

Let's verify this works as expected by executing the `run` script directly:

```
❯ ./run
Hello, World!
❯ echo $?
0
```

Notice how the exit code is 0, as expected. We're now ready to start the reduction:

```
❯ ./reduce
Reducing the following sources:
- main.jl
...
===================== done ====================
println("Hello, World!")
```

As expected, all lines except the one printing `Hello, World!` got removed.


## Notes

Test case reduction happens in parallel, so make sure there's no global effects.

When you're reducing a large project, you often will need to do some manual
editing to help the process. In that case, it can be useful to stage (`git add`)
the `depot/dev` directory to keep track of changes by C-Reduce.

If you want to reduce a test case that involves a lot of packages, ensure you run the test
case beforehand as to precompile all packages involved. The scripts in this repository use a
layered depot, making it possible to load these precompilation images during the reduction.


## Realistic example

After a refactor, CUDAnative's tests triggered an LLVM assertion when running
with `julia-debug`. To reduce this, I started by installing the necessary
packages:

```
creduce$ ./julia

julia>
] dev GPUCompiler
] dev CUDAnative
] add CUDAdrv
# the tests rely on CUDAdrv, but I'm certain the error isn't there
```

As the situation depends on running with `julia-debug`, I edited the `julia` script to use
`julia-debug` and copied the failing test into `main.jl`:

```
creduce$ ./julia main.jl
PHI node entries do not match predecessors!
julia-debug: /home/tbesard/Julia/julia/src/codegen.cpp:1511: void jl_generate_fptr(jl_code_instance_t*): Assertion `specptr != NULL' failed.
```

To trap exactly this error condition, I edit the `run` script to look for this error:

```sh
$DIR/julia main.jl |& grep "PHI node entries do not match predecessors!"
```

To speed up the search, I ran the preprocess script to get rid of comments,
whitespace, and documentation, and removed sources that are not used during
execution:

```
creduce/tools$ julia --project
] instantiate

creduce/tools$ julia --project preprocess.jl

creduce$ rm -rf depot/dev/*/{test,examples,docs,res}
```

As a final check, I ran the `run` script (as it will be run by C-Reduce)
and inspected the exit code (which initially should be 0):

```
creduce$ ./run

creduce$ echo $?
0
```

Kicking off the process, I staged all files with `git` (so that I can keep track
of C-Reduce's progress) and launched the reduce script:

```
creduce$ git add main.jl depot

creduce$ ./reduce
```

After a day or two on a system with 32 cores / 64 threads (debug builds are slow, and this
failure involved a fair number of packages and around 10.000 lines of code to reduce), all
of GPUCompiler.jl and CUDA.jl got reduced to the following:

```julia
using CUDAnative
# /home/tbesard/Julia/tools/creduce/src/50.jl

function code_llvm(a) codegen(b, a) end
# /home/tbesard/Julia/tools/creduce/src/38.jl

module a

    using GPUCompiler

        include("reflection.jl") end

# /home/tbesard/Julia/tools/creduce/src/8.jl

for a in (:code_llvm, )     @eval GPUCompiler.$a(0 )         end
# /home/tbesard/Julia/tools/creduce/src/42.jl

Base.@kwdef struct a
    b::Union{Nothing,Int}  = c
end
similar(:, d) =
    a()

# /home/tbesard/Julia/tools/creduce/src/52.jl

module GPUCompiler

include("ptx.jl")
include("driver.jl")
include("reflection.jl")
end

# /home/tbesard/Julia/tools/creduce/src/48.jl

function codegen(::Symbol, a)
    begin
        b = Dict()
        if for c in d
                e = similar(f, g)
                get(b, e) do
                 end
            end
        end
    end
end
```

To reduce this even further, I then inlined the reduced code into `main.jl`, preserving
modules and functions but removing the file structure. The error still reproduced, so this
bug isn't sensitive to the way modules are compiled and loaded. I then removed the `depot`
and kicked off another reduction that resulted with the following:

```julia
function codegen(::Symbol, a)
    begin
        b = Dict()
        if for c in d
                e = h(f, g)
                get(b, e) do
                 end
            end
        end
    end
end
Base.@kwdef struct a
    b::Union{Nothing,Int}  = c
end
h(:, d) = a()
function i(a) codegen(b, a) end
i(0)
```

After some manual clean-up, I ended up with the final reproducer:

```julia
struct a
    b::Union{Nothing,Int}
end

function main()
    c = Dict()
    d = a(undefined)
    c[d]
end

main()
```
