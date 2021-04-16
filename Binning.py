import pandas as pd
import numpy as np

small_counts = np.random.randint(0,100,20)

print(small_counts)

print(np.floor_divide(small_counts,10))

large_counts = [296, 8286, 64011, 80, 3, 725, 867, 2215, 7689, 11495, 91897, 44, 28, 7971,926, 12]

print(pd.qcut(large_counts,4,labels=None))

large_counts_series = pd.Series(large_counts)
# print(large_counts_series)
print(large_counts_series.quantile([0.25, 0.5, 0.75]))
