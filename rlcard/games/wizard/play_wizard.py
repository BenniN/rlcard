import os

import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents.human_agents.wizard_human_agent import HumanAgent
from rlcard.utils import (
    get_device,
)


def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)

    return agent


def global_payoffs(payoffs):
    for k in range(len(xglobal_payoffs)):
        xglobal_payoffs[k] += payoffs[k]
    return xglobal_payoffs


def proceed_round(numround):
    num_rounds = numround
    wizardEnv = rlcard.make(
        'wizard',
        config={
            'seed': None,
            'game_num_players': 3,
            'game_num_rounds': num_rounds,
            'game_anticipate_max_param': 0.6
        }
    )

    device = get_device()

    env_agents = []
    env_agents.append(load_model("rlcard/games/wizard/final_playing_models/model_round_" + str(num_rounds) + ".pth", wizardEnv, 0, device))
    env_agents.append(HumanAgent(num_actions=wizardEnv.num_actions))
    for _ in range(1, wizardEnv.num_players):
        env_agents.append(RandomAgent(num_actions=wizardEnv.num_actions))
        # env_agents.append(HumanAgent(num_actions=wizardEnv.num_actions))

    wizardEnv.set_agents(env_agents)

    trajectories, payoffs = wizardEnv.run(is_training=False)
    print("Trajectory", trajectories[-1][-1])
    print("Rundenpunkte: ", payoffs)
    print("gesamte Spielpunkte: ", global_payoffs(payoffs))


if __name__ == '__main__':

    '''
    Dieses File ermöglicht das Spielen eines gesamten Spiels in Wizard,
    Für jede Spielrunde eines drei Spieler-Spiels wird das passende Modell geladen. 
    Die Aggressivität der Vorhersagen ist auf 0,6 gestellt
    Der Startspieler der jeweiligen Spielrunden ist zufällig
    Es werden die Zustandsinformationen nach einem Stich mit angegeben. (dekodiert und encodiert)
    '''
    xglobal_payoffs = [0, 0, 0]
    for i in range(1, 20):
        proceed_round(i)
