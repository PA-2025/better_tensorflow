import better_tensorflow as btf
from datetime import datetime
import os

x_data = [4,5,6]
y_data =  [4,5,6]

now = datetime.now()

os.makedirs("train", exist_ok=True)

open(f"train/linear_reg_{now.strftime('%Y-%m-%d_%H-%M-%S')}", "a").close()

btf.train_linear_regression(x_data,y_data,True,100000, f"train/linear_reg_{now.strftime('%Y-%m-%d_%H-%M-%S')}")