# Code for "Robust $Q$-learning Algorithm for Markov Decision Processes under Wasserstein Uncertainty"

## Ariel Neufeld, Julian Sester

# Abstract

We present a novel $Q$-learning algorithm to solve distributionally robust Markov decision problems, where the corresponding ambiguity set of transition probabilities for the underlying Markov decision process is a Wasserstein ball around a (possibly estimated) reference measure. 
We prove convergence of the presented algorithm and provide several examples also using real data to illustrate both the tractability of our algorithm as well as the benefits of considering distributional robustness when solving stochastic optimal control problems, in particular when the estimated distributions turn out to be misspecified in practice.

# Preprint

Can be found [here](https://arxiv.org/abs/2106.10024)


# Content

The Examples from the paper are provided as seperate jupyter notebooks, each with a unique name, exactly specifying which example is covered therein. These are:
- An [Example](https://github.com/juliansester/Wasserstein-Q-learning/Example-4.1-Coin Toss.ipynb) covering Q learning for a coin toss game (Example 4.1 from the paper).
- An [Example](https://github.com/juliansester/Wasserstein-Q-learning/Example-4.2-MultiArmedBandit.ipynb) covering Q learning for self-exciting bandits (Example 4.2 from the paper).
- An [Example](https://github.com/juliansester/Wasserstein-Q-learning/Example-4.3-StockPrediction.ipynb) covering Q learning for stock price prediction (Example 4.3 from the paper).

- The file [Functions.py](https://github.com/juliansester/Wasserstein-Q-learning/Q_learning.py) contains the Python-code that is employed to train the optimal Q-learning functions.


# License

MIT License

Copyright (c) 2022 Julian Sester

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
