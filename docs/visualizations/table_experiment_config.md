# Experiment Configuration

| Case   | Model    |   Window |   Chunk | Data         |   Batch |     LR | Strategy     |   Epochs |
|:-------|:---------|---------:|--------:|:-------------|--------:|-------:|:-------------|---------:|
| Case 1 | Kosmos-2 |        8 |      10 | L+R (500)    |       1 | 0.0001 | Baseline     |       10 |
| Case 2 | Kosmos-2 |        8 |      10 | L+R (500)    |       1 | 0.0001 | Xavier Init  |       10 |
| Case 3 | Kosmos-2 |        8 |      10 | L+R (500)    |       1 | 0.0001 | Aug+Abs      |       10 |
| Case 4 | Kosmos-2 |        8 |      10 | R only (250) |       1 | 0.0001 | Baseline     |       10 |
| Case 5 | Kosmos-2 |        8 |       1 | L+R (500)    |       1 | 0.0001 | No Chunk     |        7 |
| Case 8 | Kosmos-2 |        8 |       1 | L+R (500)    |       1 | 0.0001 | No Chunk+Abs |        4 |