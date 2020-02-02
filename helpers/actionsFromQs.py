import numpy as np

# used for making help us decide actions from Qs
GAME_MAP = np.array([
    2, 4, -1, -1, -1, -1,
    1, 3, 5, -1, -1, -1,
    2, 4, 5, 6, 7, -1,
    1, 3, 7, -1, -1, -1,
    2, 3, 8, 9, -1, -1,
    3, 9, -1, -1, -1, -1,
    3, 4, 9, 10, -1, -1,
    5, 9, 11, -1, -1, -1,
    5, 6, 7, 8, 10, -1,
    7, 9, 11, -1, -1, -1,
    8, 10, -1, -1, -1, -1])

# expects a numpy array of size 72 and state of size 105
def Q_to_Actions(Qs, state):
    
    actions_for_env = []
    
    # get 7 actions
    for index in range(7):
        
        # get the max Q
        action = np.argmax(Qs)
        
        # set that Q low so we don't choose it again
        Qs[action] = -420.0
        
        # get the unit we want to map to 
        unit = action/6
        to_node = action%6
        
        # get the node we want to look at
        state_index = (unit * 5) + 45
        node = state[state_index]
        
        # compare to map
        move_to = NODE_CONNECTIONS[(node*6) + to_node]
        
        # append it to the array 
        actions_for_env.append([unit, move_to])

    return actions_for_env
    
    
