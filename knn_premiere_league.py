import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# based on https://hub.coursera-notebooks.org/user/smppgbjoljzaooewjwbuoj/notebooks/Module%201.ipynb

matches = pd.read_csv('premiere_league_total_1993-2015_preprocesado_solo_caracteristicas.csv')

print(matches.head())
