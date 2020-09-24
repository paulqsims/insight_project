# import modules
import rootpath
import pandas as pd
import numpy as np

# Set root path for project
path = rootpath.detect()

# Read in data
df = (pd.read_csv(f"{path}/data/data_clean.csv",
                  index_col=0))
