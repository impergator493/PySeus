"""

Init PyFile for Pyseus

"""

from pyseus import load
from pyseus.settings import DataType

# Full paths do not work some times, windows issue or something else?
# Sometimes Vs Code needs a restart to proper load the pictures again
# There is a difference between / and \ separator, for \ if there is a 0 after it,
# before the string there must be a r -> r'.\asdf..'

#load('../../03_Daten/fourier_data_reconstruction/prob01.h5', DataType.KSPACE)
#load(r'D:\Mario\Studium\Studieren\Masterarbeit\03_Daten\Noise_generation\Mario_Master\kspace_noise=0.2.h5', DataType.KSPACE)
#load(r'D:\Mario\Studium\Studieren\Masterarbeit\03_Daten\Noise_generation\Mario_Master\VFA_test_cart.h5', DataType.KSPACE)

load(r'D:\Mario\Studium\Studieren\Masterarbeit\03_Daten\Noise_generation\Mario_Master\image_noise=0.2_gaussian.h5', DataType.IMAGE)
#load('./tests/samples/sample.h5', DataType.IMAGE)

