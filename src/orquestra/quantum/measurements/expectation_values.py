################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
import json
from typing import Any, Dict, Iterable, List, Optional, Union, cast

import numpy as np

from ..typing import AnyPath, LoadSource
from ..utils import convert_array_to_dict, convert_dict_to_array, ensure_open
from .parities import Parities


class ExpectationValues:
    """A class representing expectation values of operators.
    For more context on how it is being used, please see the docstring of
    EstimateExpectationValues Protocol in interfaces/estimation.py.

    Args:
        values: The expectation values of a set of terms in an ising operator.
        correlations: The expectation values of pairwise products of operators.
            Contains an NxN array for each frame, where N is the number of
            operators in that frame.
        estimator_covariances: The (estimated) covariances between estimates of
            expectation values of pairs of operators. Contains an NxN array for
            each frame, where N is the number of operators in that frame. Note
            that for direct sampling, the covariance between estimates of
            expectation values is equal to the population covariance divided by
            the number of samples.

    Attributes:
        values: See Args.
        correlations: See Args.
        estimator_covariances: See Args.
    """

    def __init__(
        self,
        values: np.ndarray,
        correlations: Optional[List[np.ndarray]] = None,
        estimator_covariances: Optional[List[np.ndarray]] = None,
    ):
        self.values = values
        self.correlations = correlations
        self.estimator_covariances = estimator_covariances

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""

        data: Dict[str, Any] = {
            "frames": [],
        }  # what is "frames" for?

        data["expectation_values"] = convert_array_to_dict(self.values)

        if self.correlations:
            data["correlations"] = []
            for correlation_matrix in self.correlations:
                data["correlations"].append(convert_array_to_dict(correlation_matrix))

        if self.estimator_covariances:
            data["estimator_covariances"] = []
            for covariance_matrix in self.estimator_covariances:
                data["estimator_covariances"].append(
                    convert_array_to_dict(covariance_matrix)
                )

        return data

    @classmethod
    def from_dict(cls, dictionary: dict) -> "ExpectationValues":
        """Create an ExpectationValues object from a dictionary."""

        expectation_values = convert_dict_to_array(dictionary["expectation_values"])
        correlations: Optional[List] = None
        if dictionary.get("correlations"):
            correlations = []
            for correlation_matrix in cast(Iterable, dictionary.get("correlations")):
                correlations.append(convert_dict_to_array(correlation_matrix))

        estimator_covariances: Union[List, None] = None
        if dictionary.get("estimator_covariances"):
            estimator_covariances = []
            for covariance_matrix in cast(
                Iterable, dictionary.get("estimator_covariances")
            ):
                estimator_covariances.append(convert_dict_to_array(covariance_matrix))

        return cls(expectation_values, correlations, estimator_covariances)

    def __eq__(self, __o: object) -> bool:
        return self.__dict__ == __o.__dict__


def save_expectation_values(
    expectation_values: ExpectationValues, filename: AnyPath
) -> None:
    """Save expectation values to a file.

    Args:
        expectation_values (ExpectationValues): the expectation values to save
        file (str or file-like object): the name of the file, or a file-like object
    """
    dictionary = expectation_values.to_dict()

    with open(filename, "w") as f:
        f.write(json.dumps(dictionary, indent=2))


def load_expectation_values(file: LoadSource) -> ExpectationValues:
    """Load an array from a file.

    Args:
        file (str or file-like object): the name of the file, or a file-like object.

    Returns:
        array (numpy.array): the array
    """

    with ensure_open(file) as f:
        data = json.load(f)

    return ExpectationValues.from_dict(data)


def expectation_values_to_real(
    expectation_values: ExpectationValues,
) -> ExpectationValues:
    """Remove the imaginary parts of the expectation values

    Args:
        expectation_values (orquestra.quantum.measurement.ExpectationValues object)
    Returns:
        expectation_values (orquestra.quantum.measurement.ExpectationValues object)
    """
    values = []
    for i, value in enumerate(expectation_values.values):
        if isinstance(value, complex):
            value = value.real
        values.append(value)
    expectation_values.values = np.array(values)
    if expectation_values.correlations:
        for i, value in enumerate(expectation_values.correlations):
            if isinstance(value, complex):
                value = value.real
            expectation_values.correlations[i] = value
    return expectation_values


def concatenate_expectation_values(
    expectation_values_set: Iterable[ExpectationValues],
) -> ExpectationValues:
    """Concatenates a set of expectation values objects.

    Args:
        expectation_values_set: The expectation values objects to be concatenated.

    Returns:
        The combined expectation values.
    """

    combined_expectation_values = ExpectationValues(np.zeros(0))

    for expectation_values in expectation_values_set:
        combined_expectation_values.values = np.concatenate(
            (combined_expectation_values.values, expectation_values.values)
        )
        if expectation_values.correlations:
            if not combined_expectation_values.correlations:
                combined_expectation_values.correlations = []
            combined_expectation_values.correlations += expectation_values.correlations
        if expectation_values.estimator_covariances:
            if not combined_expectation_values.estimator_covariances:
                combined_expectation_values.estimator_covariances = []
            combined_expectation_values.estimator_covariances += (
                expectation_values.estimator_covariances
            )

    return combined_expectation_values


def get_expectation_values_from_parities(parities: Parities) -> ExpectationValues:
    """Get the expectation values of a set of operators (with precisions) from a set of
    samples (with even/odd parities) for them.

    Args:
        parities: Contains the number of samples with even and odd parities for each
            operator.

    Returns:
        Expectation values of the operators and the associated precisions.
    """
    values = []
    estimator_covariances = []

    for i in range(len(parities.values)):
        N0 = parities.values[i][0]
        N1 = parities.values[i][1]
        N = N0 + N1
        if N == 0:
            raise ValueError("There must be at least one sample for each operator")

        p = N0 / N
        value = 2.0 * p - 1.0

        # If there are enough samples and the probability of getting a sample with even
        # parity is not close to 0 or 1, then we can use p=N0/N to approximate this
        # probability and plug it into the formula for the precision.
        if N >= 100 and p >= 0.1 and p <= 0.9:
            precision = 2.0 * np.sqrt(p * (1.0 - p)) / np.sqrt(N)
        else:
            # Otherwise, p=N0/N may be not a good approximation of this probability.
            # So we use an upper bound on the precision instead.
            precision = 1.0 / np.sqrt(N)

        values.append(value)
        estimator_covariances.append(np.array([[precision**2.0]]))

    return ExpectationValues(
        values=np.array(values), estimator_covariances=estimator_covariances
    )
