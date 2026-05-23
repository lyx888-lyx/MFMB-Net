| missing | mode | mean_w_text | mean_w_audio | mean_w_vision | dominant_anchor | dominant_anchor_ratio | entropy | corr_avail_text_w_text | corr_avail_audio_w_audio | corr_avail_vision_w_vision | is_router_collapsed |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 0.1 | dynamic_soft | 0.6760 | 0.0239 | 0.3002 | text | 0.6760 | 0.1394 | 0.7395 | 0.3688 | 0.4283 | False |
| 0.1 | dynamic_soft_moe | 0.0287 | 0.3719 | 0.5995 | vision | 0.5995 | 0.4555 | 0.6097 | 0.4534 | 0.4258 | False |
| 0.2 | dynamic_soft | 0.0004 | 0.3339 | 0.6657 | vision | 0.6657 | 0.0079 | 0.5486 | 0.4465 | 0.4884 | False |
| 0.2 | dynamic_soft_moe | 0.1085 | 0.4388 | 0.4527 | vision | 0.4527 | 0.3627 | 0.6108 | 0.6170 | 0.4732 | False |
| 0.3 | dynamic_soft | 0.3333 | 0.3335 | 0.3332 | audio | 0.3335 | 0.0024 | 0.5309 | 0.5181 | 0.4662 | False |
| 0.3 | dynamic_soft_moe | 0.1985 | 0.6957 | 0.1057 | audio (audio-leaning) | 0.6957 | 0.6223 | 0.7834 | 0.7210 | 0.6497 | False |
| 0.4 | dynamic_soft | 0.6669 | 0.0011 | 0.3320 | text (text bias) | 0.6669 | 0.0145 | 0.5954 | 0.4001 | 0.4501 | False |
| 0.4 | dynamic_soft_moe | 0.0095 | 0.5399 | 0.4506 | audio (audio-leaning) | 0.5399 | 0.2542 | 0.5555 | 0.4410 | 0.4749 | False |
| 0.5 | dynamic_soft | 0.3290 | 0.3401 | 0.3310 | audio | 0.3401 | 0.0451 | 0.6304 | 0.5441 | 0.3928 | False |
| 0.5 | dynamic_soft_moe | 0.0872 | 0.5577 | 0.3551 | audio (audio-leaning) | 0.5577 | 0.2681 | 0.6054 | 0.4296 | 0.5015 | False |

Router weights show positive availability-weight correlations, indicating missing-awareness, but the dominant anchors do not always correspond to the best fixed-center method.
