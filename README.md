# orquestra-quantum

## What is it?

`orquestra-quantum` is a core library of the scientific code for [Orquestra](https://www.zapatacomputing.com/orquestra/) – the platform developed by [Zapata Computing](https://www.zapatacomputing.com) for performing computations on quantum computers.

`orquestra-quantum` provides:

- core functionalities required to run experiments, such as the `Circuit` class.
- interfaces for implementing other Orquestra modules, such as backends and optimizers.
- useful tools to support the development of workflows and other scientific projects; such as estimating observables, sampling from probability distributions, etc.

## Usage

### Workflows

Here's an example of how to use methods from `orquestra-quantum` to run a workflow. This workflow runs a circuit with a single Hadamard gate 100 times and returns the results:

```python
from orquestra.quantum.circuits import H, Circuit
from orquestra.quantum.symbolic_simulator import SymbolicSimulator

@sdk.task(
    source_import=sdk.GitImport(repo_url="git@github.com:my_username/my_repository.git", git_ref="main"),
    dependency_imports=[sdk.GitImport(repo_url="git@github.com:zapatacomputing/orquestra-quantum.git", git_ref="main")]
)
def test_orquestra_quantum():
    circ = Circuit([H(0)])
    sim = SymbolicSimulator()
    nsamples = 100
    measurements = sim.run_circuit_and_measure(circ, nsamples)
    return measurements.get_counts()


@sdk.workflow()
def test_orquestra_quantum():
    counts = test_orquestra_quantum()
    return [counts]
```

### Python

Here's an example of how to use methods from `orquestra-quantum` to run a circuit locally on python. The program runs a circuit with a single Hadamard gate 100 times and returns the results:

```python
from orquestra.quantum.circuits import H, Circuit
from orquestra.quantum.symbolic_simulator import SymbolicSimulator

circ = Circuit([H(0)])
sim = SymbolicSimulator()
nsamples = 100
measurements = sim.run_circuit_and_measure(circ, nsamples)
measurements.get_counts()
```

Even though it's intended to be used with Orquestra, `orquestra-quantum` can be also used as a standalone Python module.
To install it, you just need to run `pip install -e .` from the main directory. )

## Development and Contribution

To install the development version, run `pip install -e '.[dev]'` from the main directory. (if using MacOS, you will need single quotes around the []. If using windows, or Linux, you might not need the quotes).

We use [Google-style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) docstring format. If you'd like to specify types please use [PEP 484](https://www.python.org/dev/peps/pep-0484/) type hints instead adding them to docstrings.

There are codestyle-related [Github Actions](.github/workflows/style.yml) running for PRs, all of which should be made to `main`. 

Additionally, you can set up our [pre-commit hooks](.pre-commit-config.yaml) with `pre-commit install` . It checks for minor errors right before commiting or pushing code for quick feedback. More info [here](https://pre-commit.com). Note that if needed, you can skip these checks with the `--no-verify` option, i.e. `git commit -m "Add quickfix, prod is on fire" --no-verify`.

- If you'd like to report a bug/issue please create a new issue in this repository.
- If you'd like to contribute, please create a pull request.

### Running tests

Unit tests for this project can be run using `pytest .` from the main directory.