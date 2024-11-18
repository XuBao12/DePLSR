import pandas as pd
import os

csv = "datasets/LSDIR/data/processed1_srclip_train_lq_new.csv"
df = pd.read_csv(csv, sep="\t")
images = df["filepath"].tolist()
title = df["title"].tolist()
title2 = df["title2"].tolist()
title3 = df["title3"].tolist()
title4 = df["title4"].tolist()
title5 = df["title5"].tolist()
title6 = df["title6"].tolist()
title7 = df["title7"].tolist()



print(title3[4], title2[1], title4[5])