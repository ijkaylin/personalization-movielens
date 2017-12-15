import sys
import pytest
import pandas as pd

sys.path.append('../lib/')

import distance as dist

# data from page 34 of our textbook, using it to test similarity metrics
toy_data = pd.read_csv('../toy.dat', sep = '::', names = ['UserId', 'MovieId', 'Rating', 'Timestamp'], engine = 'python')

def test_cosine():
    # It should compute distance correctly
    def f(u, v):
        return dist.cosine_score(toy_data, u, v, by = 'user')

    assert f(1, 3) == pytest.approx(0.95, 0.01)
    assert f(2, 3) == pytest.approx(0.98, 0.01)
    assert f(3, 3) == pytest.approx(1.0, 0.01)
    assert f(4, 3) == pytest.approx(0.789, 0.01)
    assert f(5, 3) == pytest.approx(0.64, 0.01)

    # it should by symmetric
    for i in range(1, 6):
        for j in range(1, 6):
            assert f(i, j) == pytest.approx(f(j, i), 0.01)


def test_pearson():
    # It should compute distance correctly
    def f(u, v):
        return dist.pearson_score(toy_data, u, v, by = 'user')

    assert f(1, 3) == pytest.approx(0.89, 0.01)
    assert f(2, 3) == pytest.approx(0.93, 0.01)
    assert f(3, 3) == pytest.approx(0.999, 0.01)
    assert f(4, 3) == pytest.approx(-1.0, 0.01) 
    assert f(5, 3) == pytest.approx(-0.817, 0.01)

    # it should by symmetric
    for i in range(1, 6):
        for j in range(1, 6):
            assert f(i, j) == pytest.approx(f(j, i), 0.01)


