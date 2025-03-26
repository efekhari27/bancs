Hi all, 

I believe that there is an issue in the default values setup in the `ot.ResourceMap` for the `ot.BernsteinCopulaFactory`. 

When using the static method ComputeLogLikelihoodBinNumber to optimize the EBC parameter $m\in\mathbb{N}$, I am systematically getting an optimal $m^*=1$. The issue is in the definition of the default optimization bounds. By default in the `ot.ResourceMap`: $m_{max}=1$ and $m_{min} = 2$ (see the code below). 
Instead of this default bounds, for a sample with size $N\in\mathbb{N}$, I would recommend to perform an optimization between the AMISE tuning (already implemented as a static method), and the Beta copula ($m = N$).

Elias


The code: 
```
import openturns as ot 

print(f"Min value: {ot.ResourceMap.Get("BernsteinCopulaFactory-MinM")}")
print(f"Max value: {ot.ResourceMap.Get("BernsteinCopulaFactory-MaxM")}")
```
Returns: 
```
Min value: 2
Max value: 1
```

# Debbug with Joseph
There is a problem with the optimization of the EBC using the static class. 
The test sample in the cross validation should be ranked too
When the loglikelihood is applied to the border goes to infity. The fix is to modify the +1 into +0.5