kill -9 $(pgrep --parent 1 julia)

# C-Reduce for Julia

*Helper scripts to reduce Julia test cases with C-Reduce*


## Requirements

- [C-Reduce](https://embed.cs.utah.edu/creduce/) 2.8.0+, on PATH
- Julia 1.0+, on PATH


## Usage

Start by **putting your test case in `main.jl`**. If your script needs any
packages, activate the package environment by sourcing the `activate` script
from the repository's root directory, and installing packages:

```
$ cd /path/to/checkout
$ source activate
$ julia
...
```

You might want to reduce the number of Julia sources that exist in the `depot`
directory (eg. remove tests, examples, etc) to speed-up the process.

Next, **modify the `run` script** to properly catch the error you are dealing
with and return 0 if the reduced file is good. Often, you might just want to
`grep` on an error message there.

Finally, **execute the `reduce` script**. This should finalize the environment
and start C-Reduce.


## Notes

Test case reduction happens in parallel, so make sure there's no global effects.

If reducing an assertion error, you need to have built Julia with the following
options:

```
LLVM_ASSERTIONS=1
FORCE_ASSERTIONS=1
```
