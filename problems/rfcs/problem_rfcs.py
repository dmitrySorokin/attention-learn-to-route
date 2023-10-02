from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search
import numpy as np
from .lkh3_solver import solve_tsp as lkh_tsp_solver


def validate(clusters, demands, capacity, n_nodes):
    nodes = []
    for clust in clusters:
        assert len(clust) > 0
        assert sum(demands[clust]) <= capacity, f'FAIL: {clust}, {sum(demands[clust])}, {capacity}'
        nodes.extend(clust)
    assert np.all(np.asarray(sorted(nodes)) == np.arange(n_nodes)), f'FAIL: {np.asarray(sorted(nodes))}'



def split_route(route, demands, capacity, nodes):
    def dist(first, second):
        return np.sqrt(np.sum(np.square(np.asarray(first) - np.asarray(second))))
    

    n = len(route)
    V = np.ones((n, capacity + 1)) * np.infty # values
    V[n - 1] = dist(nodes[route[n - 1]], nodes[0])

    A = np.zeros((n, capacity + 1))           # actions
    to_next_action = 0
    to_depot_action = 1

    def get(i, j):
        if j > capacity:
            return np.infty
        return V[i][j]
    

    for i in reversed(range(0, n - 1)):
        for j in range(capacity + 1):
            to_next = dist(nodes[route[i]], nodes[route[i + 1]]) + get(i + 1, j + demands[route[i + 1]])
            to_depot = dist(nodes[route[i]], nodes[0]) + dist(nodes[0], nodes[route[i + 1]]) + get(i + 1, demands[route[i + 1]])
            if to_next < to_depot:
                V[i][j] = to_next
                A[i][j] = to_next_action
            else:
                V[i][j] = to_depot
                A[i][j] = to_depot_action
   
    routes = [[]]
    curr_demand = 0
    for i in range(1, n):
        curr_demand += demands[route[i]]
        routes[-1].append(route[i] - 1)
        if A[i][curr_demand] == to_depot_action and i + 1 < n:
            routes.append([])
            curr_demand = 0

    return routes, V[0][0]



class RFCS(object):

    NAME = 'rfcs'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        costs = []
        for i, tour in enumerate(pi):
            capacity = int(dataset['capacity'][i].item())
            nodes = dataset['loc'][i].tolist()
            depot = dataset['depot'][i].tolist()
            demands = dataset['demand'][i].int().tolist()

            routes, cost = split_route(
                route=[0] + (tour + 1).tolist(), 
                demands=[0] + demands, 
                capacity=capacity, 
                nodes=[depot] + nodes
            )
            validate(
                routes, 
                dataset['demand'][i].cpu().int().numpy(), 
                capacity, 
                len(dataset['loc'][i])
            )

            # costs.append(cost)

            nodes = np.asarray(nodes)
            demands = np.asarray(demands)

            result = 0
            for cluster in routes:
                if len(cluster) == 1:
                    result += 2 * np.linalg.norm(depot - nodes[cluster])
                else:
                    cost, tours = lkh_tsp_solver(depot, nodes[cluster].tolist(), demands[cluster].tolist(), capacity)
                    result += cost
            costs.append(result)

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return torch.tensor(costs), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return RFCSDataset(*args, **kwargs)

    @staticmethod
    def make_state(dataset):
        return StateTSP.initialize(dataset['loc'])

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = RFCS.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class RFCSDataset(Dataset):
    
    def __init__(self, size=100, num_samples=1000000, offset=0, seed=0, **kwargs):
        super(RFCSDataset, self).__init__()

        rnd = np.random.RandomState(seed)

        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.,
            400: 150.,
            1000: 200.,
            2000: 300., 
            5000: 300.,
            7000: 300.,
        }


        data = list(zip(
            rnd.uniform(size=(num_samples, 2)).tolist(),  # Depot location
            rnd.uniform(size=(num_samples, size, 2)).tolist(),  # Node locations
            rnd.randint(1, 10, size=(num_samples, size)).tolist(),  # Demand, uniform integer 1 ... 9
            np.full(num_samples, CAPACITIES[size]).tolist()  # Capacity, same for whole dataset
        ))

        self.data = [{
            'loc': torch.FloatTensor(loc),
            # Uniform 1 - 9, scaled by capacities
            'demand': torch.FloatTensor(demand),
            'depot': torch.FloatTensor(depot),
            'capacity': capacity,
            } for (depot, loc, demand, capacity) in data[offset:]
        ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
