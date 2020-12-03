import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_half_life(x): 
# Calculate ADF to get ADF stats and p-value  
    result = adfuller(x, maxlag=1)

    # Calculate the half-life of reversion  
    price = pd.Series(x)  
    lagged_price = price.shift(1).fillna(method="bfill")  
    delta = price - lagged_price  
    beta = np.polyfit(lagged_price, delta, 1)[0] #Use price(t-1) to predicate delta.  
    half_life = (-1*np.log(2)/beta)  
    return result[0], result[1], half_life
