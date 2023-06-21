# AdaGCL

This is the PyTorch implementation for AdaptiveGCL proposed in the paper **Adaptive Graph Contrastive Learning for Recommendation**.

## 1. Running environment

We develop our codes in the following environment:

- python==3.9.13
- numpy==1.23.1
- torch==1.11.0
- scipy==1.9.1

## 2. Datasets

| Dataset      | # User | # Item | # Interaction | Interaction Density |
| ------------ | ------ | ------ | ------------- | ------------------- |
| Last.FM      | 1,892  | 17,632 | 92,834        | 2.8 × $10^{-3}$     |
| Yelp         | 42,712 | 26,822 | 182,357       | 1.6 × $10^{-4}$     |
| BeerAdvocate | 10,456 | 13,845 | 1,381,094     | 9.5 × $10^{-3}$     |

## 3. How to run the codes

- Last.FM

```python
python Main.py --data lastfm --gamma -0.95 --ib_reg 1e-2
```

- Yelp

```python
python Main.py --data yelp --ssl_reg 1 --ib_reg 1e-2 --epoch 100
```

- BeerAdvocate

```python
python Main.py --data beer --ib_reg 1e-2 --lambda0 1e-2 --ssl_reg 1
```

