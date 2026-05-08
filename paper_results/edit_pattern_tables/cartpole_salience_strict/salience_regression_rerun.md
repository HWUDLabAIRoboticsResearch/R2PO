# Salience-Problem Regression Proxy Rerun

- Episodes analyzed: 1000
- Worst-reference detector: `word`
- Denominator for headline proxy: 304 regressions (`delta < 0`)

| Criterion | Count | % of Regressions |
| --- | ---: | ---: |
| delta < 0, worst < best, and worst <= median | 237 | 78.0% |
| delta < 0, worst < median | 237 | 78.0% |
| old reward proxy plus Critic-LLM explicitly references the worst rollout/trajectory | 233 | 76.6% |
| strict reward proxy plus Critic-LLM explicitly references the worst rollout/trajectory | 233 | 76.6% |

The strict `worst < median` change excludes 0 old-proxy cases (0.0% of old-proxy cases) where `worst = median < best`.

## Per Environment

| Env | Regressions | Old | Strict | Old + Worst Ref | Strict + Worst Ref |
| --- | ---: | ---: | ---: | ---: | ---: |
| cartpole | 304 | 237 (78.0%) | 237 (78.0%) | 233 (76.6%) | 233 (76.6%) |
