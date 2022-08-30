from rlcard.games.wizard.utils import create_wizard_dmc_graph

if __name__ == '__main__':
    model_path = 'final_dmc_models/dmc_5e06_round1'
    create_wizard_dmc_graph(model_path)