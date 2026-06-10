# Bedrock recon yield table

pass@1 = mean correctness over gens; pass@16 = fraction of problems with >=1 correct gen

## deepseek_v3p2  (calls=2357, errors=0)

| model | category | n_prob | n_gen | pass@1 | pass@16 | tok_p50 |
|---|---|---|---|---|---|---|
| deepseek_v3p2 | cryptarithm_deduce | 148 | 2357 | 0.003 | 0.027 | 2500 |

## glm5  (calls=1243, errors=0)

| model | category | n_prob | n_gen | pass@1 | pass@16 | tok_p50 |
|---|---|---|---|---|---|---|
| glm5 | cryptarithm_deduce | 78 | 1243 | 0.001 | 0.013 | 3182 |

## gptoss_120b  (calls=9376, errors=0)

| model | category | n_prob | n_gen | pass@1 | pass@16 | tok_p50 |
|---|---|---|---|---|---|---|
| gptoss_120b | bit_manipulation | 150 | 2400 | 0.169 | 0.533 | 1257 |
| gptoss_120b | cryptarithm_deduce | 150 | 2400 | 0.000 | 0.007 | 682 |
| gptoss_120b | cryptarithm_guess | 150 | 2400 | 0.004 | 0.027 | 678 |
| gptoss_120b | equation_numeric_guess | 136 | 2176 | 0.091 | 0.206 | 842 |

## gptoss_120b_val  (calls=3, errors=0)

| model | category | n_prob | n_gen | pass@1 | pass@16 | tok_p50 |
|---|---|---|---|---|---|---|
| gptoss_120b_val | cryptarithm_deduce | 2 | 3 | 0.000 | 0.000 | 32000 |

## kimi_k2p5  (calls=1337, errors=0)

| model | category | n_prob | n_gen | pass@1 | pass@16 | tok_p50 |
|---|---|---|---|---|---|---|
| kimi_k2p5 | cryptarithm_deduce | 84 | 1337 | 0.001 | 0.012 | 7380 |

## mistral_large3  (calls=9376, errors=0)

| model | category | n_prob | n_gen | pass@1 | pass@16 | tok_p50 |
|---|---|---|---|---|---|---|
| mistral_large3 | bit_manipulation | 150 | 2400 | 0.175 | 0.560 | 5199 |
| mistral_large3 | cryptarithm_deduce | 150 | 2400 | 0.002 | 0.007 | 2200 |
| mistral_large3 | cryptarithm_guess | 150 | 2400 | 0.000 | 0.007 | 2170 |
| mistral_large3 | equation_numeric_guess | 136 | 2176 | 0.055 | 0.176 | 3269 |

## nemotron_super3_120b  (calls=1513, errors=0)

| model | category | n_prob | n_gen | pass@1 | pass@16 | tok_p50 |
|---|---|---|---|---|---|---|
| nemotron_super3_120b | cryptarithm_deduce | 95 | 1513 | 0.000 | 0.000 | 8192 |

## qwen3_235b  (calls=864, errors=0)

| model | category | n_prob | n_gen | pass@1 | pass@16 | tok_p50 |
|---|---|---|---|---|---|---|
| qwen3_235b | cryptarithm_deduce | 54 | 864 | 0.001 | 0.019 | 7281 |

