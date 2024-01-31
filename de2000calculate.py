import numpy as np
from pyciede2000 import ciede2000
import math

def xyz2lab(xyzs):
    labs=np.zeros(3)
    Xn = 573
    Yn = 609
    Zn = 706

    if xyzs[0] / Xn > math.pow(24 / 116, 3):
        fX = math.pow(xyzs[0] / Xn, 1 / 3)
    else:
        fX = (841 / 108) * (xyzs[0] / Xn) + 16 / 116
    if xyzs[1] / Yn > math.pow(24 / 116, 3):
        fY = math.pow(xyzs[1] / Yn, 1 / 3)
    else:
        fY = (841 / 108) * (xyzs[1] / Yn) + 16 / 116
    if xyzs[2] / Zn > math.pow(24 / 116, 3):
        fZ = math.pow(xyzs[2] / Zn, 1 / 3)
    else:
        fZ = (841 / 108) * (xyzs[2] / Zn) + 16 / 116
    labs[0] = 116 * fY - 16
    labs[1] = 500 * (fX - fY)
    labs[2] = 200 * (fY - fZ)
    return labs

xyzpre = np.loadtxt('xyzpre.txt')
xyzgt = np.loadtxt('xyzgt.txt')
eab = np.zeros([xyzgt.shape[0]])
for i in range(xyzgt.shape[0]):
    res = ciede2000(xyz2lab(xyzpre[i]), xyz2lab(xyzgt[i]))
    eab[i] = res['delta_E_00']

# max_index = np.argmax(eab)
# eab = np.delete(eab,max_index)

print(eab)
print(eab.max())
print(eab.min())
print(eab.mean())