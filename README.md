## RLT
RLT: A reinforcement learning approach to test mobile games.

This is a research prototype automatic game testing.

## Installation
Simply clone the source code from this repository and apply the follwing enviroment configuration


## Enviroment Configration
* PYTHON
* ANDROID SDK
* LINUX
* EMULATOR OR DEVICES
* Candy Crush Game

##USAGE

$python train_candy.py
$python test_candy.py

##Apply data augmentation

1) flip horizontally
	Uncomment the lines 105-108 in file gym_candy/gym_candy/envs/agent.py
	#flip-lr
    state_new = np.flip(state, 1)
    next_state_new = np.flip(next_state, 1)
    action_new = self.flip_lr(action)
    self.memory.append((state_new, action_new, reward, next_state_new, done))
        
2) flip vertically
	Uncomment the lines 115-121 in file gym_candy/gym_candy/envs/agent.py	
	#flip-up
    if(np.array_equal(state[33:164,11:137,:],next_state[33:164,11:137,:])):
        print('state is equal to the next state')
	
	    state_new = np.flip(state[33:164,11:137,:], 0)
	    next_state_new = np.flip(next_state[33:164,11:137,:], 0)
	    action_new = self.flip_up(action)
	    self.memory.append((state_new, action_new, reward, next_state_new, done))
Then, run the training codes and test codes sequentially.
	$python train_candy.py
	$python test_candy.py
