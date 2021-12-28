import pandas as pd
import pylab as plt

# Create dataframe
file_name = "epochs_1000_model.log"
df = pd.read_csv(file_name)
df.plot()
plt.show()
