# Final-Dissertation
MS Dissertation in UoS, UK.
## v4.0 Description
### Running
First add following commands to `/gym/envs/__init__.py`

```python
register(
    id="RobotInterception-v1",
    entry_point="gym.envs.final:DiscreteRobotMovingEnv",
    max_episode_steps=10000000,
)

register(
    id="RobotInterception-v2",
    entry_point="gym.envs.final:TwoWheelRobotMovingEnv",
    max_episode_steps=10000000,
)
register(
    id="RobotInterception-v3",
    entry_point="gym.envs.final:DiscreteRobotTranslationEnv",
    max_episode_steps=10000000,
)

register(
    id="RobotInterception-v4",
    entry_point="gym.envs.final:TwoWheelRobotContinuousMovingEnv",
    max_episode_steps=10000000,
)
```

Then, moving the `/final` folder to `/gym/envs/` directory.

Run following command in terminal.

``` bash
python main_discrete.py --max_episode=1000 --max_iterations=1000
```
