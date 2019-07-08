[![PyPI version](https://badge.fury.io/py/sos-fsharp.svg)](https://badge.fury.io/py/sos-fsharp)

# sos-fsharp
SoS extension for F# using [IfSharp kernel](https://github.com/fsprojects/IfSharp). Please refer to the [SoS homepage](http://vatlab.github.io/SOS) and [kernel development for SoS](https://vatlab.github.io/sos-docs/doc/user_guide/language_module.html) for more information.

## Run: Example development workflow


```
git clone https://github.com/vatlab/sos-fsharp.git
cd sos-fsharp/
python3.6 -m venv sosvenv
source sosvenv/bin/activate
pip install wheel
pip install git+https://github.com/vatlab/sos-notebook.git
pip install -e .

```

You'll also need to install [ifsharp](https://github.com/fsprojects/IfSharp) for the F# kernel.



If removal of pip editable installation is required, do `rm -r $(find . -name '*.egg-info')` from `sos-fsharp` when venv is enabled.


## Tests

Install [Chrome](https://www.google.com/chrome/browser/) and [ChromeDriver](https://chromedriver.storage.googleapis.com/index.html?path=75.0.3770.90/) to run the tests.

You must `source setenv.sh` before running the tests.

```
pip install selenium pytest requests nose
pytest .
```

