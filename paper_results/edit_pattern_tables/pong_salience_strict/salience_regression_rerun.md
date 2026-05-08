# Salience-Problem Regression Proxy Rerun

- Episodes analyzed: 1000
- Worst-reference detector: `word`
- Denominator for headline proxy: 323 regressions (`delta < 0`)

| Criterion | Count | % of Regressions |
| --- | ---: | ---: |
| delta < 0, worst < best, and worst <= median | 265 | 82.0% |
| delta < 0, worst < median | 102 | 31.6% |
| old reward proxy plus Critic-LLM explicitly references the worst rollout/trajectory | 261 | 80.8% |
| strict reward proxy plus Critic-LLM explicitly references the worst rollout/trajectory | 101 | 31.3% |

The strict `worst < median` change excludes 163 old-proxy cases (61.5% of old-proxy cases) where `worst = median < best`.

## Per Environment

| Env | Regressions | Old | Strict | Old + Worst Ref | Strict + Worst Ref |
| --- | ---: | ---: | ---: | ---: | ---: |
| pong | 323 | 265 (82.0%) | 102 (31.6%) | 261 (80.8%) | 101 (31.3%) |
