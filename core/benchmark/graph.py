import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("core/benchmark/benchmark.csv")

# add a column with the total time
df["total_time"] = df["png_conversion_time"] + df["prediction_time"] + df["gsps_creation_time"] + df["gsps_sending_time"] + df["tmp_files_clearing_time"]
#plot if their is a correlation between the number of bboxes or size and the total time
sns.pairplot(df,vars=["nb_bboxes","size","total_time"])
plt.show()

#plot an histogram of the total time
plt.hist(df["total_time"],bins=30)
plt.xlabel("Total time (s)")
plt.ylabel("Number of files")
plt.title("Histogram of the total time")
plt.show()

# plot the average time for each step
plt.bar(["png_conversion_time","prediction_time","gsps_creation_time","gsps_sending_time","tmp_files_clearing_time"],[df["png_conversion_time"].mean(),df["prediction_time"].mean(),df["gsps_creation_time"].mean(),df["gsps_sending_time"].mean(),df["tmp_files_clearing_time"].mean()])
plt.xlabel("Step")
plt.ylabel("Average time (s)")
plt.title("Average time for each step")
plt.show()

# plot the average time for each step with the standard deviation
plt.bar(["png_conversion_time","prediction_time","gsps_creation_time","gsps_sending_time","tmp_files_clearing_time"],[df["png_conversion_time"].mean(),df["prediction_time"].mean(),df["gsps_creation_time"].mean(),df["gsps_sending_time"].mean(),df["tmp_files_clearing_time"].mean()],yerr=[df["png_conversion_time"].std(),df["prediction_time"].std(),df["gsps_creation_time"].std(),df["gsps_sending_time"].std(),df["tmp_files_clearing_time"].std()])
plt.xlabel("Step")
plt.ylabel("Average time (s)")
plt.title("Average time for each step")
plt.show()


