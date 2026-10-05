Build
-----
You can build the docs locally on your machine. From this `docs` directory, run a clean
build with the `docs` extra (which also bundles the `pandoc` binary needed to render the
example notebooks):
```console
uv run --extra docs make build
```
View the result by opening `_build/html/index.html` in a web browser.

`make build` clears the generated API pages (`source/_autosummary`), the copied example
notebooks (`source/_examples`), and `_build` before building. `make html` builds
incrementally instead, and `make clean` only clears.

The example notebooks come from a clone of
[slmsuite-examples](https://github.com/holodyne/slmsuite-examples) beside this repository
(`../slmsuite-examples` from the repository root) if one exists, and are otherwise
downloaded from GitHub.
