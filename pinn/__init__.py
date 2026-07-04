"""Physical layer of the Crusoe factory digital twin: Multi-Head Physics-Informed NN.

One model per monitored machine (framed as a tire curing/vulcanization press,
Michelin-inspired). Shared recurrent body + one output head per physical
phenomenon:

- vibration head      -> trained on CWRU + NASA IMS (real public data)
- thermal/power head  -> trained on AI4I 2020 (synthetic-but-documented public data)
- degradation/RUL head-> trained on NASA C-MAPSS FD001 (simulated public data)
- pressure head       -> trained on SYNTHETIC curing-cycle data generated in-repo
                         (no public dataset exists for curing-press pressure;
                         clearly labeled synthetic everywhere it appears)

No confidential Michelin data is used anywhere. Public datasets serve as
physically-plausible stand-ins and are always labeled as such.
"""

__version__ = "0.1.0"
