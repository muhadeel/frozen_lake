# Frozen Lake

## Project Structure

The project is written in python, the main directory called `frozen_lake`, the following tree structure shows the code organization.

```
frozen_lake
├── environments
│   └── __init__.py
│   └── base_environment.py
│   └── frozen_lake_environment.py
├── rl_algorithms
│   └── __init__.py
│   └── non_tabular_model_free.py
│   └── tabular_model_based.py
│   └── tabular_model_free.py
└── .gitignore
└── main.py
└── README.md
```

- The first package `environments` contains 3 python modules:
    1. A python package initializer `__init__.py` (empty).
    2. A python module `base_environment.py` which contains the two super classes; `EnvironmentModel` and `Environment`.
`Environment` is a subclass of `EnvironmentModel`.
    3. A python module `frozen_lake_environment.py` which contains `FrozenLake` class. `FrozenLake` is our implementation of the frozen lake environment (*the first task*), it is a subclass of `Environment` class. 


- The second package `rl_algorithms` contains reinforcement learning methods:
    1. A python package initializer `__init__.py` (empty).
    2. A python module `non_tabular_model_free.py`, it aims to solve *the fourth task* `Non-tabular model-free reinforcement learning`.
    3. A python module `tabular_model_based.py`, it aims to solve *the second task* `Tabular model-based reinforcement learning`.
    4. A python module `tabular_model_free.py`, it aims to solve *the third task* `Tabular model-free reinforcement learning`.


- The python script `main.py` is an implementation of running the frozen lake environment using all the implemented reinforcement learning methods with two possible versions **Small Lake and Big Lake**.


- README.md, a read me file for this project.

**Our implementation did not deviate from what was suggested**

## Usage
Execute the python script from the root folder (**frozen_lake**): `python -m main` 
```
usage: python -m main [-h] [-l {small,big}]

optional arguments:
  -h, --help            show this help message and exit
  -l {small,big}, --lake {small,big}
                        Choose which lake to run: `small` for small lake OR `big` for big lake, default: small
```
The python script takes one argument `lake`, the value could be `small` to run small frozen lake:
```
python -m main -l small
```
or big, to run big frozen lake:
```
python -m main -l big
```

The default behaviour is `small`

