Delta vs fixed text baseline (same missing rate).

| missing | method | delta_MAE | delta_Corr | delta_Non0_acc_2 | delta_Non0_F1_score |
|---:|---|---:|---:|---:|---:|
| 0.1 | text | +0.0000 = same | +0.0000 = same | +0.0000 = same | +0.0000 = same |
|  | audio | -0.0075 ↑ improved | +0.0008 ↑ improved | -0.0031 ↓ worse | -0.0020 ↓ worse |
|  | vision | -0.0423 ↑ improved | +0.0075 ↑ improved | -0.0005 ↓ worse | -0.0009 ↓ worse |
|  | **dynamic_soft** | -0.0276 ↑ improved | +0.0047 ↑ improved | -0.0005 ↓ worse | -0.0007 ↓ worse |
|  | **dynamic_soft_moe** | +0.0051 ↓ worse | -0.0136 ↓ worse | -0.0168 ↓ worse | -0.0166 ↓ worse |
| 0.2 | text | +0.0000 = same | +0.0000 = same | +0.0000 = same | +0.0000 = same |
|  | audio | -0.0917 ↑ improved | +0.0184 ↑ improved | +0.0142 ↑ improved | +0.0143 ↑ improved |
|  | vision | -0.0893 ↑ improved | +0.0150 ↑ improved | +0.0270 ↑ improved | +0.0276 ↑ improved |
|  | **dynamic_soft** | +0.0094 ↓ worse | -0.0172 ↓ worse | +0.0087 ↑ improved | +0.0090 ↑ improved |
|  | **dynamic_soft_moe** | +0.0226 ↓ worse | -0.0081 ↓ worse | -0.0066 ↓ worse | -0.0060 ↓ worse |
| 0.3 | text | +0.0000 = same | +0.0000 = same | +0.0000 = same | +0.0000 = same |
|  | audio | +0.0076 ↓ worse | -0.0053 ↓ worse | +0.0157 ↑ improved | +0.0123 ↑ improved |
|  | vision | -0.1046 ↑ improved | +0.0035 ↑ improved | +0.0356 ↑ improved | +0.0339 ↑ improved |
|  | **dynamic_soft** | -0.0161 ↑ improved | -0.0114 ↓ worse | +0.0117 ↑ improved | +0.0061 ↑ improved |
|  | **dynamic_soft_moe** | -0.0762 ↑ improved | -0.0019 ↓ worse | +0.0203 ↑ improved | +0.0163 ↑ improved |
| 0.4 | text | +0.0000 = same | +0.0000 = same | +0.0000 = same | +0.0000 = same |
|  | audio | +0.0288 ↓ worse | -0.0163 ↓ worse | +0.0061 ↑ improved | +0.0056 ↑ improved |
|  | vision | +0.1161 ↓ worse | -0.0256 ↓ worse | -0.0030 ↓ worse | +0.0003 ↑ improved |
|  | **dynamic_soft** | -0.0137 ↑ improved | +0.0027 ↑ improved | +0.0214 ↑ improved | +0.0206 ↑ improved |
|  | **dynamic_soft_moe** | +0.0537 ↓ worse | -0.0233 ↓ worse | +0.0092 ↑ improved | +0.0166 ↑ improved |
| 0.5 | text | +0.0000 = same | +0.0000 = same | +0.0000 = same | +0.0000 = same |
|  | audio | +0.0183 ↓ worse | -0.0087 ↓ worse | -0.0168 ↓ worse | -0.0142 ↓ worse |
|  | vision | -0.0098 ↑ improved | -0.0028 ↓ worse | +0.0081 ↑ improved | +0.0095 ↑ improved |
|  | **dynamic_soft** | +0.0395 ↓ worse | -0.0859 ↓ worse | -0.0386 ↓ worse | -0.0257 ↓ worse |
|  | **dynamic_soft_moe** | +0.0893 ↓ worse | -0.0157 ↓ worse | -0.0503 ↓ worse | -0.0091 ↓ worse |
