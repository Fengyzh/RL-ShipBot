# Versions
 - nQL.py and testQ.py are all for testing, you can play around with it
 - DQN3.py is what I'm working on for DQN


# Use Partial vision in Q table learning

## How to implement:
    - Default Q learning uses the state to store things (exp: Q-table: [12, [1,4,7,2]])
        - Where the 12 is the state, usually the posiiton in the 2D array
        - [1,4,7,2] is the Q value of the action in that state
        - Q learning works by picking the highest Q value action to take at the state the agent is currently in
    
    - With partial vision, we want to store things like (exp: Q-table: [[0,0,-1,1,5,-1,0,10,-1], [1,4,7,2]])
        - The [0,0,-1,1,5,-1,0,10,-1] represent the surrounding space
        - [1,4,7,2] is the Q value of the action in that state
        - Now the agent no longer depends on the environment but depends on the surrounding information instead

# Finished Tasks
    - DQN5.py is a working DQN copy that uses experience replay
        - It has no partial visibility and it takes the whole world as the input state to the NN
    - QLearn.py currently in Beta, training is working