# C-Reduce for Julia

*Helper scripts to reduce Julia test cases with C-Reduce*


## Requirements

- [C-Reduce](https://embed.cs.utah.edu/creduce/) 2.8.0+, on PATH
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

Packages that you `add` will be ignored by the reduction process; only packages
you `dev` will take part in it. However, all Julia code in those packages will
be considered, so you might want to reduce the amount of it to speed up the
reduction process (e.g. remove tests, examples, etc) to speed-up the process.

Next, **modify the `run` script** to properly catch the error you are dealing
with and return 0 if the reduced file is good. Often, you want to look for
specific output in the standard error stream.

Finally, **execute the `reduce` script**. This should finalize the environment
and start C-Reduce.


## Notes

Test case reduction happens in parallel, so make sure there's no global effects.

When you're reducing a large project, you often will need to do some manual
editing to help the process. In that case, it can be useful to stage (`git add`)
the `depot/dev` directory to keep track of changes by C-Reduce.

To use a different build of Julia, e.g. with assertions enabled or using a
sanitizer, change the invocation in the `run` script.
