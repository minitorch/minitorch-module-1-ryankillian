from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """
    `ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes:
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


class Scalar:
    """
    A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        """Add two Scalars together using the Add scalar function."""
        if isinstance(b, Scalar):
            return Add.apply(self, b)
        else:
            return Add.apply(self, Scalar(b))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Return 1.0 if self is less than b, otherwise 0.0."""
        if isinstance(b, Scalar):
            return LT.apply(self, b)
        else:
            return LT.apply(self, Scalar(b))

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Return 1.0 if self is greater than b, otherwise 0.0."""
        if isinstance(b, Scalar):
            return LT.apply(b, self)
        else:
            return LT.apply(Scalar(b), self)

    def __eq__(self, b: ScalarLike) -> Scalar:  # type: ignore[override]
        return EQ.apply(self, b)

    def __hash__(self) -> int:
        # Use the unique identifier for hashing if available.
        return hash(self.unique_id)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Subtract another Scalar from this one by adding the negative of the second Scalar."""
        if isinstance(b, Scalar):
            return Add.apply(self, Neg.apply(b))
        else:
            return Add.apply(self, Neg.apply(Scalar(b)))

    def __neg__(self) -> Scalar:
        """Return the negative of this scalar."""
        return Neg.apply(self)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    def log(self) -> Scalar:
        """Return the natural logarithm of this scalar."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Return the exponential of this scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Return the sigmoid function applied to this scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Return the ReLU function applied to this scalar."""
        return ReLU.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x: value to be accumulated
        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        "True if this variable created by the user (no `last_fn`)"
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """
        Compute the gradients for each input variable of the last function applied to this variable
        using the chain rule of calculus. This is a part of the backward pass in automatic differentiation.

        Args:
            d_output (Any): The derivative of the output with respect to some upstream gradient.
                            This is typically the gradient from a later part of the network or the final output.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple contains a variable
                                            and the gradient of the output with respect to that variable.

        Details:
            This function uses the last function (`last_fn`) that was applied to create this variable,
            invokes its `backward` method with the current context (`ctx`) and the passed derivative (`d_output`).
            This effectively applies the chain rule, where the derivative of the function with respect to each
            of its inputs is computed and scaled by `d_output`.

            The function ensures compatibility with functions returning both single values and tuples
            by normalizing the output of `backward` to a tuple. This makes the subsequent zipping with
            `inputs` straightforward and error-free.
        """

        # Ensure the variable has a history object with necessary attributes to perform backpropagation.
        h = self.history
        assert h is not None, "History must not be None"
        assert h.last_fn is not None, "Last function applied must not be None"
        assert h.ctx is not None, "Context of the last function must not be None"

        # Call the backward method of the last function with the current derivative
        derivs = h.last_fn.backward(h.ctx, d_output)  # type: ignore

        # Normalize the derivatives to a tuple if not already one
        if type(derivs) is not tuple:
            derivs = (derivs,)

        # Pair each input variable with its corresponding derivative
        return zip(h.inputs, derivs)

    def backward(self, d_output: Optional[float] = None) -> None:
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """
    Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters:
        f : function from n-scalars to 1-scalar.
        *scalars  : n input scalar values.
    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)  # type: ignore
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
