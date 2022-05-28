import torch

def make_11action(env, action_index, always_attack=True):
    # Action들을 정의
    action = env.action_space.noop()
    # Cameras
    if (action_index == 0):
        action['camera'] = [0, 0]
    elif (action_index == 1):
        action['camera'] = [0, -5]
    elif (action_index == 2):
        action['camera'] = [0, 5]
    elif (action_index == 3):
        action['camera'] = [-5, 0]
    elif (action_index == 4):
        action['camera'] = [5, 0]

    # Forwards
    elif (action_index == 5):
        action['forward'] = 0
    elif (action_index == 6):
        action['forward'] = 1

    # Jump
    elif (action_index == 7):
        action['jump'] = 0
    elif (action_index == 8):
        action['jump'] = 1

    # Attack 
    elif (action_index == 9):
        action['attack'] = 0
    elif (action_index == 10):
        action['attack'] = 1
    
    return action

def save_model(episode, SAVE_PERIOD, SAVE_PATH, model, MODEL_NAME, ENV_NAME):
    if episode % SAVE_PERIOD == 0:
        save_path_name = SAVE_PATH + ENV_NAME+'_'+MODEL_NAME+'_'+str(episode)+'.pt'
        torch.save(model.state_dict(), save_path_name)
        print("model saved")

def converter(observation, device):
    obs = observation['pov']
    obs = obs / 255.0
    obs = torch.from_numpy(obs)
    obs = obs.permute(2, 0, 1)
    return obs.float().to(device)
