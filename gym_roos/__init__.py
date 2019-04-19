from gym.envs.registration import register

register(
    id='SaccadeDigit-v0',
    entry_point='gym_roos.envs:EnvSaccadeDigit',
)

register(
    id='SaccadeMultDigits-v0',
    entry_point='gym_roos.envs:EnvSaccadeMultDigits',
)
