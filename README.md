# MDVO
This is the source code of paper, named "Mean-field Learning for Decentralized Task Offloading in Vehicular Edge Computing", and the proposed solution and comparison algorithms are implemented.

## Environment
- Python 3.9
- Tensorflow 2.12.0

## Algorithms
- Mean-field reinforcement learning based Decentralized Vehicular edge task Offloading (MDVO): our proposed task offloading scheme. `MDVO/mdvo_vec.py`
- A Cooperative Multi-agent Deep Reinforcement Learning (CMADRL) scheme [1]:it is a decentralized dynamic computation offloading method based on Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm , which is a quite popular multi-agent DRL algorithm. Due to the action space and reward function of [1] are different from MDVO, we have only adopted the MADDPG algorithm from [1] to learn computation offloading policies. The simulation setup and task offloading framework are the same with MDVO. `CMADRL/experiments/main.py`
- A centralized Service Offloading (SOL) scheme [2]:It is a digital twin-empowered centralized task offloading scheme for vehicular edge computing, which is based on the Double Deep Q-learning Network (DDQN) algorithm. We have designed a central edge server in our system, and precisely reproduced SOL with the same simulation setup of MDVO. `SOL/SOL_VEC.py`
- Double Deep Q-learning Network (DDQN) scheme: In this method, we retained the MDVO's framework and adopted the DDQN algorithm from [2] to generate offloading decisions. Although both utilizing DDQN based algorithm, this method is a decentralized task offloading scheme, while SOL is a centralized method.`DDQN/ddqn_de_vec.py`
- Random offloading (RO) scheme: In this scheme, MDVO’s framework is retained, vehicles randomly choose the offloading ratio and target edge server for their new tasks. `MDVO/random.py`

## References
- [1]Z. Chen, L. Zhang, Y. Pei, C. Jiang, and L. Yin, “Noma based multi-user mobile edge computation offloading via cooperative multi-agent deep reinforcement learning,” IEEE Trans. Cogn. Commun. Netw., vol. 8, no. 1, pp. 350-364, 2022. https://doi.org/10.1109/TCCN.2021.3093436.
- [2]X.Xu, B. Shen, S. Ding, G. Srivastava, M. Bilal, M. R. Khosravi,V. G. Menon, M. A. Jan, and M. Wang, “Service offloading with deep Q-network for digital twinning-empowered Internet of Vehicles in edge computing,” IEEE Trans.Ind.Inform., vol. 18, pp. 1414-1423, 2022. https://doi.org/10.1109/TII.2020.3040180.
