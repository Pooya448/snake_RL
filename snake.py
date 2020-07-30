import numpy as np
from ple import PLE
from ple.games.snake import Snake


def round_state(state):
    newState = {}
    for i in state:
        if i != 'snake_body' and i != 'snake_body_pos':
            newState[i] = (int(state[i])/60)
    return newState

agent = Snake(width=256, height=256)


env = PLE(agent, fps=15, force_fps=False, display_screen=True)

env.init()

actions = env.getActionSet()

q_table = {}
alpha = 0.1
gamma = 0.9


while True:

    print(q_table)

    old_game_state = round_state(agent.getGameState())

    if env.game_over():
        env.reset_game()

    up = q_table.get(tuple(old_game_state.values()) + (119,), 0)
    right = q_table.get(tuple(old_game_state.values()) + (97,), 0)
    left = q_table.get(tuple(old_game_state.values()) + (100,), 0)
    down = q_table.get(tuple(old_game_state.values()) + (115,), 0)

    list = [up, right, left, down]
    max_act = max(list)

    counter = 0

    if max_act == up:
        action = 119
        counter += 1
    if max_act == right:
        action  = 97
        counter += 1
    if max_act == left:
        action = 100
        counter += 1
    if max_act == 115:
        action = 115
        counter += 1
    if counter >= 2:
        action = actions[np.random.randint(0, len(actions))]

    reward = env.act(action)

    new_game_state = round_state(agent.getGameState())
    #
    old_Q_state = tuple(old_game_state.values()) + (action,)

    if old_Q_state not in q_table.keys():
        q_table[old_Q_state] = 0

    next_up = q_table.get(tuple(new_game_state.values()) + (119,), 0)
    next_right = q_table.get(tuple(new_game_state.values()) + (97,), 0)
    next_left = q_table.get(tuple(new_game_state.values()) + (100,), 0)
    next_down = q_table.get(tuple(new_game_state.values()) + (115,), 0)

    next_max = max(next_up, next_right, next_left, next_down)
    sample = reward + gamma * next_max
    q_table[old_Q_state] = (1 - alpha) * q_table[old_Q_state] + alpha * sample

    counter_for_zeros = 0
    counter_all = 0
    for key in q_table:
        counter_all += 1
        if q_table[key] == 0:
            counter_for_zeros +=1
    print(float(counter_for_zeros) / float(counter_all) * 100)
