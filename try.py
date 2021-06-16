import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np

a = 2

x = np.linspace(stats.gamma.ppf(0.01, a), stats.gamma.ppf(0.99, a), 100)

b = stats.gamma.rvs(a=a, size=2000)


z = pd.DataFrame(stats.gamma.interval(alpha=0.95, a=b.mean(), scale=stats.gamma.std(b)))

z = z.transpose()

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15,8))

ax[0].plot(x, stats.gamma.pdf(x, a))
ax[0].hist(b, density=True, histtype='stepfilled', alpha=0.2)

ax[1].plot(b)
ax[1].plot(z[z.columns[0]])
ax[1].plot(z[z.columns[1]])
plt.show()
