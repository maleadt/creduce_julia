# C-Reduce for Julia

*Helper scripts to reduce Julia test cases with C-Reduce*

## Set-up

* Install [C-Reduce](https://embed.cs.utah.edu/creduce/) 2.8.0+, and make it
  discoverable in `PATH`.
* Put the test-case in `src/`: it can consist of multiple files, but no
  directories. The `src/` directory will be added to the `LOAD_PATH` so you can
  recreate a (file-based) package hierarchy there.
* Edit the `run` script to return success upon the expected failure, and make
  sure it executes properly. 

Finally, execute the `reduce` script.


## Notes

Test case reduction happens in parallel, so make sure there's no global effects.

If reducing an assertion error, you need to have built Julia with the following
options:

```
LLVM_ASSERTIONS=1
FORCE_ASSERTIONS=1
```

For a greatly improved user experience, we would need something similar to `gcc
-E` that makes Julia spit out all processed sources for each (pre)compilation
job, eg. with expanded `include`s ready to be fed to C-Reduce.
