# C-Reduce for Julia

*Helper scripts to reduce Julia test cases with C-Reduce*

## Set-up

* Install [C-Reduce](https://embed.cs.utah.edu/creduce/) 2.8.0+, and make it
  discoverable in `PATH`.
* Put the test-case in `src/`: it can consist of multiple files and directories,
  but names of files that are to be reduced (ie. Julia sources) have to be
  unique. The `src/` directory will be added to the `LOAD_PATH` so you can
  recreate a package hierarchy there.
* Edit the `run` script to return success upon the expected failure, and make
  sure it executes properly. 

Finally, execute the `reduce` script.


## Demo

The repository contains a sample error in an imported package. Run `reduce` to
see it get reduced:

```
$ ./reduce                                                                                                                                                                                              *[master] 
===< 29066 >===
running 4 interestingness tests in parallel
...
===================== done ====================

pass statistics:
  method pass_clex :: rename-toks worked 1 times and failed 17 times
  method pass_balanced :: parens-to-zero worked 2 times and failed 3 times
  method pass_indent :: final worked 3 times and failed 0 times
  method pass_blank :: 0 worked 3 times and failed 0 times
  method pass_clex :: rm-toks-1 worked 3 times and failed 22 times
  method pass_indent :: regular worked 4 times and failed 2 times
  method pass_lines :: 10 worked 5 times and failed 18 times
  method pass_lines :: 1 worked 5 times and failed 18 times
  method pass_lines :: 8 worked 5 times and failed 18 times
  method pass_lines :: 4 worked 5 times and failed 18 times
  method pass_lines :: 2 worked 5 times and failed 18 times
  method pass_lines :: 3 worked 5 times and failed 18 times
  method pass_lines :: 6 worked 5 times and failed 18 times
  method pass_lines :: 0 worked 6 times and failed 18 times

          ******** src/Foo.jl ********

using Bar
          ******** src/main.jl ********

using Foo
          ******** src/Bar/src/Bar.jl ********

module a error("example error message") end
```


## Notes

Test case reduction happens in parallel, so make sure there's no global effects.

If reducing an assertion error, you need to have built Julia with the following
options:

```
LLVM_ASSERTIONS=1
FORCE_ASSERTIONS=1
```
