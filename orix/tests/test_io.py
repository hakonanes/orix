import pytest
import numpy as np
import os

from orix import io


@pytest.fixture(
    params=[
        """4.485496 0.952426 0.791507     0.000     0.000   22.2  0.060  1       6
1.343904 0.276111 0.825890    19.000     0.000   16.3  0.020  1       2""",
    ]
)
def angfile(tmpdir, request):
    f = tmpdir.mkdir("angfiles").join("angfile.ang")
    f.write(
        """# File created from ACOM RES results
# ni-dislocations.res
#
#
# MaterialName      Nickel
# Formula
# Symmetry          43
# LatticeConstants  3.520  3.520  3.520  90.000  90.000  90.000
# NumberFamilies    4
# hklFamilies       1  1  1 1 0.000000
# hklFamilies       2  0  0 1 0.000000
# hklFamilies       2  2  0 1 0.000000
# hklFamilies       3  1  1 1 0.000000
#
# GRID: SqrGrid#"""
    )
    f.write(request.param)
    return str(f)


@pytest.mark.parametrize(
    "angfile, expected_data",
    [
        (
            """4.485496 0.952426 0.791507     0.000     0.000   22.2  0.060  1       6
1.343904 0.276111 0.825890    19.000     0.000   16.3  0.020  1       2
1.343904 0.276111 0.825890    38.000     0.000   18.5  0.030  1       3
1.343904 0.276111 0.825890    57.000     0.000   17.0  0.060  1       6
4.555309 2.895152 3.972020    76.000     0.000   20.5  0.020  1       2
1.361357 0.276111 0.825890    95.000     0.000   16.3  0.010  1       1
4.485496 0.220784 0.810182   114.000     0.000   20.5  0.010  1       1
0.959931 2.369110 4.058938   133.000     0.000   16.5  0.030  1       3
0.959931 2.369110 4.058938   152.000     0.000   16.1  0.030  1       3
4.485496 0.220784 0.810182   171.000     0.000   17.4  0.020  1       2""",
            np.array(
                [
                    [0.77861956, 0.12501022, -0.44104243, -0.42849224],
                    [-0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [-0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [-0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [0.05331986, -0.95051048, -0.28534763, 0.11074093],
                    [-0.45489991, -0.13271448, -0.03640618, -0.87984517],
                    [0.8752001, 0.02905178, -0.10626836, -0.47104969],
                    [0.3039118, -0.01972273, 0.92612154, -0.22259272],
                    [0.3039118, -0.01972273, 0.92612154, -0.22259272],
                    [0.8752001, 0.02905178, -0.10626836, -0.47104969],
                ]
            ),
        ),
    ],
    indirect=["angfile"],
)
def test_load_ang(angfile, expected_data):
    loaded_data = io.loadang(angfile)
    assert np.allclose(loaded_data.data, expected_data)


def test_load_ctf():
    """ Crude test of the ctf loader """
    z = np.random.rand(100, 8)
    np.savetxt("temp.ctf", z)
    z_loaded = io.loadctf("temp.ctf")
    os.remove("temp.ctf")
