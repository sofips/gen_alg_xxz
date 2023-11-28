# gen_alg_xxz
Implementation of genetic algorithms to optimize transmission fidelity in quantum spin chains described by XXZ hamiltonians.


## Physical Description

We consider spin chains described by XXZ Hamiltonians:

$$
H\ = \sum _{i=1}^{N} J_i \left( \sigma _{i}^{x} \sigma _{i+1}^{x} +\sigma _{i}^{y} \sigma _{i+1}^{y} + \Delta \sigma _{i}^{z} \sigma _{i+1}^{z}\right)
$$

with n being the number of elements in the chain. Our goal is to implement genetic algorithms in order to find optimal values for the coupling constants $J_i$. In this case, optimal values mean values that maximize the fidelity when a quantum state is being transmitted through the chain. 

Since refelection symmetry is a neccessary condition to obtain perfect transmission, only half of the coupling values are calculated. For example, if the chain has 21 elements, there will be 20 coupling constants $J_i$ and the algorithm will maximize for 10 variables. 

This program is designed to optimize chains with an **even number of couplings** to simplify the reflection of values.

To find a detailed explanation of the physical problem, check [1] or [2]. 

## References
[1] - G. M. Nikolopoulos and I. Jex, Quantum State Transfer and Network Engineering. Berlin Springer, 2016

[2] - X.M. Zhang, Z.W. Cui, X. Wang, and M.H. Yung, “Automatic spin-chain learning to explore the quantum speed limit”
