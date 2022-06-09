from rlcard.games.cego.utils import create_cego_dmc_graph

if __name__ == '__main__':
    model_path = 'experiments/cego_dmc_standard'
    create_cego_dmc_graph(model_path)