from rlcard.games.wizard.utils import create_wizard_dmc_graph

if __name__ == '__main__':
    model_path = 'final_models/dmc_cego'
    create_wizard_dmc_graph(model_path)