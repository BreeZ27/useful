import pandas as pd
import numpy as np


# Вычисляет дисперсию вектор-столбца
def dispersion(x):
    z = len(x) - 1
    x = x - x.values.mean()
    return np.sum(np.square(x)) / z
