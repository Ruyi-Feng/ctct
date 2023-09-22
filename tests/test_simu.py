from simu import Simu
from simu.params import params



def test_simu():
    args = params()
    vsl = Simu(args)
    vsl.run(aim_rtg=410)

test_simu()
