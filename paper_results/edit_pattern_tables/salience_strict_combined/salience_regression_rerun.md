# Salience-Problem Regression Proxy Rerun

- Episodes analyzed: 2000
- Worst-reference detector: `word`
- Denominator for headline proxy: 627 regressions (`delta < 0`)

| Criterion | Count | % of Regressions |
| --- | ---: | ---: |
| delta < 0, worst < best, and worst <= median | 502 | 80.1% |
| delta < 0, worst < median | 339 | 54.1% |
| old reward proxy plus Critic-LLM explicitly references the worst rollout/trajectory | 494 | 78.8% |
| strict reward proxy plus Critic-LLM explicitly references the worst rollout/trajectory | 334 | 53.3% |

The strict `worst < median` change excludes 163 old-proxy cases (32.5% of old-proxy cases) where `worst = median < best`.

## Per Environment

| Env | Regressions | Old | Strict | Old + Worst Ref | Strict + Worst Ref |
| --- | ---: | ---: | ---: | ---: | ---: |
| cartpole | 304 | 237 (78.0%) | 237 (78.0%) | 233 (76.6%) | 233 (76.6%) |
| pong | 323 | 265 (82.0%) | 102 (31.6%) | 261 (80.8%) | 101 (31.3%) |
