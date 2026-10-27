"""The fraud simulator."""

import pytest

from qrc.datasets.create_dataset import (
    _n_compromised_on_day,
    add_frauds,
    generate_dataset,
    scaled_simulation,
)


@pytest.fixture(scope="module")
def simulation():
    # enough customers that a few compromised per day stay a small share
    return generate_dataset(n_customers=300, n_terminals=600, nb_days=60, default_random_state=0)


def scenario_3_frauds(simulation, **kwargs):
    customers, terminals, transactions = simulation
    transactions = add_frauds(customers, terminals, transactions.copy(), **kwargs)
    return (transactions.TX_FRAUD_SCENARIO == 3).sum()


def test_default_compromises_three_customers_per_day(simulation):
    assert scenario_3_frauds(simulation) == scenario_3_frauds(simulation, n_compromised_customers=3)


def test_more_compromised_customers_give_more_scenario_3_frauds(simulation):
    assert scenario_3_frauds(simulation, n_compromised_customers=9) > 2 * scenario_3_frauds(simulation)


def test_integer_compromise_rates_compromise_that_many_every_day():
    assert all(_n_compromised_on_day(3, day) == 3 for day in range(365))
    assert all(_n_compromised_on_day(3.0, day) == 3 for day in range(365))


def test_fractional_compromise_rates_spread_over_the_days():
    counts = [_n_compromised_on_day(0.25, day) for day in range(365)]
    assert set(counts) == {0, 1}
    assert sum(counts) == 91


def test_full_size_scaling_is_the_handbook_simulation():
    simulation, frauds = scaled_simulation(5000)
    assert simulation == dict(n_customers=5000, n_terminals=10000, r=5)
    assert frauds == dict(n_compromised_terminals=2, n_compromised_customers=3)


@pytest.mark.parametrize("n_customers", [25, 250])
def test_scaled_simulations_keep_the_handbook_fraud_rate(n_customers):
    simulation, frauds = scaled_simulation(n_customers)
    customers, terminals, transactions = generate_dataset(**simulation, default_random_state=0)
    transactions = add_frauds(customers, terminals, transactions, **frauds)
    # the shared dataset has 0.86% frauds, all three scenarios present
    assert 0.005 < transactions.TX_FRAUD.mean() < 0.015
    assert set(transactions.TX_FRAUD_SCENARIO) == {0, 1, 2, 3}
