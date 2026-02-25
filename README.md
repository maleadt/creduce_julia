# Julia source-code reduction

This repository contains a set of scripts to reduce Julia test cases with
[cvise](https://github.com/marxin/cvise). It can be helpful if you want to
minimize code that causes a crash or bug, without having to iteratively edit a
script and manually verify the bug still occurs.

Note that testcase reduction works best if the testcase can be evaluated quickly
(i.e., seconds rather than minutes). If your testcase takes a long time to run,
you may want to do some manual editing first to get rid of dependencies that
take a long time to load.


## Requirements

- [cvise](https://github.com/marxin/cvise), on PATH
- Julia 1.10+, on PATH


## Usage

### Simple case (just `main.jl`, no packages to reduce)

1. Put your failing code in `repro/main.jl`.

2. Edit `julia` for the right Julia version/build/flags.

3. If packages are needed, install them:

   ```
   ./julia -e 'using Pkg; Pkg.add("SomePackage")'
   ```

4. Verify that `./julia repro/main.jl` reproduces the error.

5. Edit `run` to grep for the specific error pattern, verifying that
   `./run` returns 0 if the error is present, and non-zero otherwise.

6. Run `./reduce` to start reduction.


### Complex case (reducing packages too)

In addition to the above:

1. Copy sources of dependencies you want to reduce into `repro/`:

   ```
   cp -r ~/.julia/packages/ForwardDiff/xxx/ repro/ForwardDiff
   ```

2. Edit `repro/Project.toml` with `[sources]` entries:

   ```toml
   [deps]
   ForwardDiff = "f6369f11-..."

   [sources]
   ForwardDiff = {path = "ForwardDiff"}
   ```

   Resolve the environment if needed.

3. (Optionally) Get rid of unnecessary files in the copied package sources, e.g., `test/`, `docs/`, `examples/`, `src/` (keep only the files that are actually used by your test case).

4. (Optionally) Preprocess to strip comments, docstrings, and whitespace:

   ```
   julia --project=tools tools/preprocess.jl
   ```


## Demo

Let's start with a simple test case that throws an "interesting error":

```
cat > repro/main.jl << 'EOF'
println("Something unrelated")
error("Interesting error")
EOF
```

We can verify this executes correctly:

```
❯ ./julia --project=repro repro/main.jl
Something unrelated
ERROR: LoadError: Interesting error
```

Since we are only interested in the error, we adapt the `run` script to grep for
it:

```sh
$DIR/julia --project=repro -L $DIR/setup.jl repro/main.jl 2>&1 | grep "Interesting error"
```

Let's verify this works as expected:

```
❯ ./run
ERROR: LoadError: Interesting error
❯ echo $?
0
```

Now start the reduction:

```
❯ ./reduce
Reducing the following sources:
  repro/main.jl
...
===================== done ====================
println("Hello, World!")
```

As expected, all lines except the one printing `Hello, World!` got removed.


## Notes

Test case reduction happens in parallel, so make sure there are no global
effects.

When reducing a large project, you often will need to do some manual editing to
help the process. In that case, it can be useful to stage (`git add`) the
`repro/` directory to keep track of changes by cvise.

If you want to reduce a test case that involves many packages, ensure you run
the test case beforehand to precompile all packages involved. The scripts in
this repository use a layered depot, making it possible to load these
precompilation images during the reduction.
