from typing import List, Dict, Union

def generate_exchange_graph(exchanges: List[str]) -> Dict[str, Dict[str,  Dict[str, Union[str, float]]]]:
    """
    Creates an undirected graph from list of exchanges. 
    Currencies and exchanges are represented as vertices and edges respectively.
    Graph also contains latest ask/bid for each exchange.

    The graph is represented as nested dicts. Each key of each dict is a vertex,
    each leaf node is the exhange symbol for the vertices leading to that leaf.

    example:
    self.currencies = ['USD', 'ETH', 'BTC']
    self.exchanges = ['BTC-USD', 'ETH-USD', 'ETH-BTC']
    
    graph = {
        'USD': {
            'BTC': {
                'ccy': 'BTC-USD',
                'ask': 0.2345,
                'bid': 1.234,
            },
            'ETH': {
                'ccy': 'ETH-USD',
                'ask': 0.2345,
                'bid': 1.234,
            }
        }
        'BTC': {
            'USD': {
                'ccy': 'BTC-USD',
                'ask': 0.2345,
                'bid': 1.234,
            },
            'ETH': {
                'ccy': 'ETH-BTC',
                'ask': 0.2345,
                'bid': 1.234,
            },
        }
        'ETH': {
            'BTC': {
                'ccy': 'ETH-BTC',
                'ask': 0.2345,
                'bid': 1.234,
            },
            'USD': {
                'ccy': 'ETH-USD',
                'ask': 0.2345,
                'bid': 1.234,
            },
        }
    }

    :return: (Dict[str, Dict[str,  Dict[str, Union[str, float]]]]) 
             currency exchange graph
    """
    currencies = set()
    for ccy in exchanges:
        base, asset = ccy.split('-')
        currencies.add(base)
        currencies.add(asset)
    currencies = list(currencies)
    graph = {}
    for vertex in currencies:
        edges = {}
        for edge in exchanges:
            edge_ends = edge.split('-')
            if vertex in edge_ends:
                idx = int(not bool(edge_ends.index(vertex)))
                edges[edge_ends[idx]] = {
                    'ccy': edge,
                    'ask': 0.0,
                    'bid': 0.0,
                }
        graph[vertex] = edges
    return graph