# score_based_par_est
This Matlab code is related to our paper "Score-Based Parameter Estimation for a Class of Continuous-Time State Space Models" published in 
SIAM Journal on Scientific Computing (click [here](https://epubs.siam.org/doi/abs/10.1137/20M1362942) to view the paper.)
The paper is concerned with estimating unknown parameters in a continuous-time hidden Markov model (state space model). This code is used to generate Figure 5 
in the published article for model 1. The code for he rest of other models is very similar.


Let d<sub>x</sub> = d<sub>y</sub> = 1, d<sub>&theta;</sub> = 2 and consider the following linear SDE:
     
dX<sub>t</sub> = &theta;<sub>1</sub> X<sub>t</sub>  dt + &sigma;  dW<sub>t</sub>

dY<sub>t</sub> = &theta;<sub>2</sub> (&kappa; - X<sub>t</sub>)  dt +  dB<sub>t</sub>,

where {W<sub>t</sub>}<sub>t>=0</sub> and {B<sub>t</sub>}<sub>t>=0</sub> are two independent Brownian motions. &kappa; and
&sigma; are both fixed and &theta;<sub>1</sub>, &theta;<sub>2</sub> are the parameters to be inferred.
