# Environment details

This project uses the Unity platform to train two agents to play tennis. In this case, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of −0.01. Thus, the goal of each agent is to keep the ball in play.  

The observation space consists of eight variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  

Two continuous actions are available, corresponding to movement toward or away from the net, and jumping. The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 over 100 consecutive episodes to considered the environment as solved.  

# Learning algorithm

The reinforcement learning algorithm being used in this project is a multiagent deep deterministic policy gradients, or MADDPG. DDPG combines the strengths of policy-based (stochastic) and value-based (deterministic) AI learning methods by using two agents, called the Actor and the Critic. The actor directly estimates the optimal policy, or action, for a given state, and applies gradient ascent to maximize rewards. The critic takes the actor's output and uses it to estimate the value (or cumulative future reward) of state-action pairs. The weights of the actor are then updated with the critic’s output, and the critic is updated with the gradients from the temporal-difference error signal at each step. This hybrid algorithm can be a very robust form of artificial intelligence, because it needs fewer training samples than a purely policy-based agent, and demonstrates more stable learning than a purely value-based one.

The multiagent DDPG combines several agents, in this case two, and trains them using the same neural networks and replay buffer, the agents share their experiences for increasing learning speed.

# Important considerations

In order to achieve an stable and reliable training of the agent some important considerations are taking into account:

* Modifying the Agent.step() method to accommodate multiple agents, and to employ a learning interval. This ensures that the agent performs the learning step only once every 10 time steps during training, and each time 2 passes are made through experience sampling and the Agent.learn() method:

  ``` def step(self, state, action, reward, next_state, done, count):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn every x steps (in this case 10), if enough samples are available in memory, and learn 5 times in that moment.
        if len(self.memory) > BATCH_SIZE:
            if count % 10 == 0:
                for i in range(0,2):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA) ```

* In the OUNoise.sample() method, changing random.random() to np.random.standard_normal(). This means that the random noise being added to the experience replay buffer samples via the Ornstein-Uhlenbeck process follows a Gaussian distribution, and turns out to perform much better than a completely random distribution in this case!

  ``` dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size) ```

# Model architecture and hyperparameters

## Architecture for the Actor

Fully connected layer 1: Input 8 (state space), Output 512, RELU activation.    
Batch normalization layer 1: Input 512 and output 512. Smooth training.  
Fully connected layer 2: Input 512, Output 256, RELU activation.  
Fully connected layer 3: Input 256, Output 3 (action space), tanh activation for clipping output between -1 and 1.   

## Architecture for the Critic

Fully connected layer 1: Input 8 (state space), Output 512, RELU activation.    
Batch normalization layer 1: Input 512 and output 512. Smooth training.  
Fully connected layer 2: Input 512, Output 256, RELU activation.    
Fully connected layer 3: Input 256, Output 3 (action space).     

The hyperparameters for tweaking and optimizing the learning algorithm were:

max_t (30000): maximum number of timesteps per episode.      
BUFFER_SIZE = int(1e6)  replay buffer size  
BATCH_SIZE = 128        minibatch size  
GAMMA = 0.99            discount factor  
TAU = 1e-3              for soft update of target parameters  
LR_ACTOR = 1e-4         learning rate of the actor   
LR_CRITIC = 1e-4        learning rate of the critic  
WEIGHT_DECAY = 0        L2 weight decay  
GRAD_CLIPPING = 0       Activate gradient clippin in critic.    

# Plot of rewards

Environment solved in 8879 episodes!	Average Score: 0.51
The plot of rewards for this run is as follows:
<img width="365" height="236" alt="Trend_solucionDDPGTennis" src="https://github.com/user-attachments/assets/00c6ae5b-5216-4d22-9166-3a1a5a9bd1aa" />



# Future work

It might be useful to experiment with other network architectures for this project - different numbers of hidden layers, different numbers of nodes, and additional features such as dropout.

Increasing the size of the experience replay buffer had a major effect on the performance of the agent, and it might perform better with an even larger buffer. It might also be useful to try implementing prioritized experience replay, instead of a random buffer.

I would like also experimenting with competitive agents instead of collaborating ones, that would mean to create different agent instances for the actors in order to have a competitive play, changes in the rewards should be necessary too, as the reward should be for winning points not only for hittin the ball.

# References

* [DDPG](https://arxiv.org/abs/1509.02971)
* [MADDPG](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf)
