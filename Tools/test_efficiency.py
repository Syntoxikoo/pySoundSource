import exemple.sound_processing as sound_processing
import sig_proc_tools
from time import time
import numpy as np


mag = 20
mag2 = np.random.rand(2000) * 10
start = time()
db_int1 = sound_processing.mag_to_db(mag)
print("time to run with method 1 for int :", time() - start)

start = time()
db_int2 = sig_proc_tools.mag2db(mag)
print("time to run with method 2 for int :", time() - start)

start = time()
db_int1 = sound_processing.mag_to_db(mag2)
print("time to run with method 1 for array :", time() - start)

start = time()
db_int2 = sig_proc_tools.mag2db(mag2)
print("time to run with method 2 for array :", time() - start)
