![GitHub actions status](https://github.com/rgal/gym-2048/actions/workflows/python-package.yml/badge.svg)
[![Coverage Status](https://codecov.io/gh/rgal/gym-2048/branch/master/graph/badge.svg)](https://codecov.io/gh/rgal/gym-2048)


# gym-2048

Open AI gym environment for the game 2048 and agents to play it.

## Getting Started

This project contains an Open AI gym environment for the game 2048 (in directory gym-2048) and some agents and tools to learn to play it.

* `simple_rl_agent.py` - Simple RL agent which learns the quality of moves for a particular board state
* `train_deep_network.py` - Trains a deep neural network to play from SL data
* `gather_training_data.py` - records training data to train neural network from in CSV form

Other training_data files are for manipulating the training data.

### Prerequisites

What things you need to install the software and how to install them

For gym environment:
```
gym
numpy
```

For agents and recording training data:

```
pygame
tensorflow
```

For running tests:

```
pytest
```

### Installing

You can get set up to develop this project with:

```
python setup.py develop
```

## Running the tests

You can run the unit tests for this project by running:

```
py.test
```

## Contributing

Please report issues or create pull requests if you want to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

