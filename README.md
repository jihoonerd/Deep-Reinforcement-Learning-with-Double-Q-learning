# Deep Reinforcement Learning with Double Q-learning

![atlantis_playing](/assets/atlantis.gif)

This repository implements the paper: **[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)**.

The authors of the paper applied [Double Q-learning](https://papers.nips.cc/paper/3964-double-q-learning) concept on their DQN algorithm. This paper proposed Double DQN, which is similar to DQN but more robust to overestimation of Q-values.

The major difference between those two algorithms is the way to calculate Q-value from target network. Compared to the DQN, directly using Q-value from target network, DDQN chooses an action that maximizes the Q-value of main network at the next state.

#### DQN
![dqn_y_target](/assets/y_dqn.png)

#### DDQN
![ddqn_y_target](/assets/y_ddqn.png)

Most of the implementation is almost the same as the [implementation of DQN](https://github.com/jihoonerd/Human-level-control-through-deep-reinforcement-learning).

## Features

* Employed ***TensorFlow 2*** with performance optimization
* Simple structure
* Easy to reproduce

## Model Structure

![nn.svg](/assets/nn.svg)

## Requirements

***Default running environment is assumed to be CPU-ONLY. If you want to run this repo on GPU machine, just replace `tensorflow` to `tensorflow-gpu` in package lists.***

## How to install

### `virtualenv`

```bash
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## How to run

You can run Atari 2600 game with `main.py`. Running environment needs to be `NoFrameskip` from `gym` package.

```bash
$ python main.py --help
usage: main.py [-h] [--env ENV] [--train] [--play PLAY]
               [--log_interval LOG_INTERVAL]
               [--save_weight_interval SAVE_WEIGHT_INTERVAL]

Atari: DQN
optional arguments:
  -h, --help            show this help message and exit
  --env ENV             Should be NoFrameskip environment
  --train               Train agent with given environment
  --play PLAY           Play with a given weight directory
  --log_interval LOG_INTERVAL
                        Interval of logging stdout
  --save_weight_interval SAVE_WEIGHT_INTERVAL
                        Interval of saving weights
```

### Example 1: Train BreakoutNoFrameskip-v4

``` bash
$ python main.py --env BreakoutNoFrameskip-v4 --train
```

### Example 2: Play PongNoFrameskip-v4 with trained weights

```bash
$ python main.py --env PongNoFrameskip-v4 --play ./log/[LOGDIR]/weights
```

### Example 3: Control log & save interval

```bash
$ python main.py --env BreakoutNoFrameskip-v4 --train --log_interval 100 --save_weight_interval 1000
```

## Results

This implementation is guaranteed to work well for `Atlantis`, `Boxing`, `Breakout` and `Pong`. Tensorboard summary is located at `./archive`. Tensorboard will show following information:

* Average Q value
* Epsilon (for exploration)
* Latest 100 avg reward (clipped)
* Loss
* Reward (clipped)
* Test score
* Total frames

```bash
$ tensorboard --logdir=./archive/
```

Single RTX 2080 Ti is used for the results below. (Thanks to [@JKeun](https://github.com/JKeun) for allowing his computation resources)

### Atalntis

* Orange: DQN
* Blue: DDQN

#### Reward

![atlantis](/assets/atlantis_result.png)

#### Q-value

![atlantis_Q](/assets/DDQN_Q-value.png)

We can see that DDQN's average Q-value is suppressed compared to that of DQN.

## BibTeX

```
@article{hasselt2015doubledqn,
  abstract = {The popular Q-learning algorithm is known to overestimate action values under
certain conditions. It was not previously known whether, in practice, such
overestimations are common, whether they harm performance, and whether they can
generally be prevented. In this paper, we answer all these questions
affirmatively. In particular, we first show that the recent DQN algorithm,
which combines Q-learning with a deep neural network, suffers from substantial
overestimations in some games in the Atari 2600 domain. We then show that the
idea behind the Double Q-learning algorithm, which was introduced in a tabular
setting, can be generalized to work with large-scale function approximation. We
propose a specific adaptation to the DQN algorithm and show that the resulting
algorithm not only reduces the observed overestimations, as hypothesized, but
that this also leads to much better performance on several games.},
  added-at = {2019-11-18T11:40:13.000+0100},
  author = {van Hasselt, Hado and Guez, Arthur and Silver, David},
  biburl = {https://www.bibsonomy.org/bibtex/2c2bad4b4c5a34cb31a3f569c71e851ab/jan.hofmann1},
  description = {[1509.06461] Deep Reinforcement Learning with Double Q-learning},
  interhash = {d3061c37961afb78096e314854dd90bc},
  intrahash = {c2bad4b4c5a34cb31a3f569c71e851ab},
  keywords = {dqn q-learning reinforcement_learning},
  note = {cite arxiv:1509.06461Comment: AAAI 2016},
  timestamp = {2019-11-18T11:40:13.000+0100},
  title = {Deep Reinforcement Learning with Double Q-learning},
  url = {http://arxiv.org/abs/1509.06461},
  year = 2015
}
```

## Author
Jihoon Kim ([@jihoonerd](https://github.com/jihoonerd))
