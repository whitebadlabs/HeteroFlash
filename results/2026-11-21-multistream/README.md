# Several users at once (GLM-5.3-Flash, top-4 + union + mean stand-in)

One boot, the Max-Q expert server healthy throughout. Each run: N concurrent streams, 8 prompts x 512 tokens, greedy,
fixed length. Per setting about 100 runs (a looping driver repeated its second round; every run is a valid sample).
"Share" = fraction of each decode window's distinct experts computed on the Spark GPU (the balance rule A); "cap k" =
at most k GPU experts per layer-window.

| users | setting | runs | total tok/s mean (min-max) |
|---|---|---|---|
| 1 | share 0.50 | 114 | 36.7 (35.6-37.8) |
| 2 | share 0.20 | 110 | 43.4 (41.5-45.8) |
| 2 | share 0.25 | 115 | 44.6 (42.4-47.0) |
| 2 | share 0.30 | 117 | 46.2 (43.3-48.7) |
| 2 | share 0.35 | 109 | 47.6 (46.0-49.5) |
| 2 | cap 5 | 95 | 44.6 (43.0-46.9) |
| 4 | share 0.10 | 107 | 48.9 (46.5-51.8) |
| 4 | share 0.15 | 98 | 50.5 (47.7-53.6) |
| 4 | share 0.20 / 0.25 | 1 / 1 | 52.0 / 49.2 (too few runs) |
| 4 | cap 5 / cap 7 | 1 / 2 | 50.7 / 52.4 (too few runs) |

Earlier run with the single-user split (share 0.50), total tok/s for 1 / 2 / 4 users: top-5 34.8 / 34.9 / 24.5,
top-4 37.8 / 39.6 / 28.2. Tool calling (tool-eval-bench, leaderboard settings, 4 in parallel) for top-4: 89/100
(top-5 90, all-experts baseline 89).
