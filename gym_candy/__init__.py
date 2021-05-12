from gym.envs.registration import register
  
register(id='Candy-v0',
    entry_point='gym_candy.envs:CandyEnv')
