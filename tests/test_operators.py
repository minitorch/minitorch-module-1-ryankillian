from typing import Callable, List, Tuple

import pytest
from hypothesis import given
from hypothesis.strategies import lists

from minitorch import MathTest
from minitorch.operators import (
    add,
    addLists,
    eq,
    id,
    inv,
    inv_back,
    is_close,
    log_back,
    lt,
    max,
    mul,
    neg,
    negList,
    prod,
    relu,
    relu_back,
    sigmoid,
    sum,
)

from .strategies import assert_close, small_floats

# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x: float, y: float) -> None:
    "Check that the main operators all return the same value of the python version"
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if abs(x) > 1e-5:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a: float) -> None:
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a: float, b: float) -> None:
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a: float) -> None:
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a: float) -> None:
    "Check that a - 1.0 is always less than a"
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a: float) -> None:
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a: float) -> None:
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    """
    Check properties of the sigmoid function, specifically:
    * It is always between 0.0 and 1.0.
    * One minus sigmoid is the same as sigmoid of the negative (-a).
    * It is strictly increasing.

    Args:
        a (float): Input to the sigmoid function.
    """
    sigmoid_a = sigmoid(a)
    sigmoid_neg_a = sigmoid(-a)

    # It is always between 0.0 and 1.0.
    assert 0.0 <= sigmoid_a <= 1.0, "Sigmoid output should be within [0, 1]"

    # One minus sigmoid(a) is the same as sigmoid of the negative (-a).
    assert (
        abs((1.0 - sigmoid_a) - sigmoid_neg_a) < 1e-7
    ), "One minus sigmoid(a) should equal sigmoid(-a)"

    # Ensure it is strictly increasing by comparing two points
    sigmoid_a_plus = sigmoid(a + 1e-5)  # a small increment to compare increase
    assert sigmoid_a_plus >= sigmoid_a or is_close(
        sigmoid_a_plus, sigmoid_a
    ), "Sigmoid should be strictly increasing from a to a + 1e-5"


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    """
    Test the transitive property of less-than using the `lt` function.

    This test checks that if `lt(a, b)` and `lt(b, c)` both indicate true (1.0),
    then `lt(a, c)` should also indicate true (1.0).

    Args:
        a (float): First floating-point number.
        b (float): Second floating-point number.
        c (float): Third floating-point number.
    """
    if lt(a, b) == 1.0 and lt(b, c) == 1.0:
        assert (
            lt(a, c) == 1.0
        ), "Transitive property failed (lt(a, b) and lt(b, c) should imply lt(a, c))"


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(a: float, b: float) -> None:
    """
    Test that multiplication is symmetric, meaning that the order of operands does not change the result.

    Args:
        a (float): First operand.
        b (float): Second operand.
    """
    assert mul(a, b) == mul(
        b, a
    ), "Multiplication failed to be symmetric with inputs {}, {}".format(a, b)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x: float, y: float, z: float) -> None:
    """
    Test the distributive property of multiplication over addition.

    This test checks the distributive property, specifically verifying that:
    z * (x + y) == z * x + z * y

    Args:
        x (float): First operand for addition.
        y (float): Second operand for addition.
        z (float): Operand for multiplication.
    """
    # Compute both sides of the distributive property
    left_side = mul(z, add(x, y))
    right_side = add(mul(z, x), mul(z, y))

    # Assert that both sides are equal
    assert is_close(
        left_side, right_side
    ), f"Distributive property failed for values x={x}, y={y}, z={z}"


@pytest.mark.task0_2
@given(small_floats)
def test_multiplicative_identity(a: float) -> None:
    """
    Test the multiplicative identity property of the multiplication function.

    This test checks that any number multiplied by 1 returns the number itself, validating:
    a * 1 = a

    Args:
        a (float): A floating-point number to test the multiplicative identity property.
    """
    # Multiplying by 1 should return the number itself
    assert mul(a, 1) == a, f"Multiplicative identity failed for a = {a}"


# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a: float, b: float, c: float, d: float) -> None:
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    """
    Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    # Calculate the sum of each list individually
    sum_ls1 = sum(ls1)
    sum_ls2 = sum(ls2)

    # Calculate the sum of both lists together
    combined_list_sum = sum(addLists(ls1, ls2))

    # Assert the sum of sums is equal to the sum of the combined elements
    assert is_close(sum_ls1 + sum_ls2, combined_list_sum), (
        f"Expected the sum of individual sums ({sum_ls1} + {sum_ls2}) to be equal to "
        f"the sum of combined elements ({combined_list_sum})"
    )


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls: List[float]) -> None:
    assert_close(sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x: float, y: float, z: float) -> None:
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls: List[float]) -> None:
    check = negList(ls)
    for i, j in zip(ls, check):
        assert_close(i, -j)


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn: Tuple[str, Callable[[float], float]], t1: float) -> None:
    name, base_fn = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(
    fn: Tuple[str, Callable[[float, float], float]], t1: float, t2: float
) -> None:
    name, base_fn = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a: float, b: float) -> None:
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)
