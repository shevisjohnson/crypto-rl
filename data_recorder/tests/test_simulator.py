from datetime import datetime as dt

from joblib import Parallel, parallel_backend, delayed

from configurations import TIMEZONE, BASKET, EXCHANGES
from data_recorder.database.simulator import Simulator


def test_get_tick_history() -> None:
    """
    Test case to query Arctic TickStore
    """
    start_time = dt.now(tz=TIMEZONE)

    sim = Simulator()
    query = {
        'ccy': ['BTC-USD'],
        'start_date': 20181231,
        'end_date': 20190102
    }
    tick_history = sim.db.get_tick_history(query=query)
    print('\n{}\n'.format(tick_history))

    elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


def test_get_orderbook_snapshot_history() -> None:
    """
    Test case to export testing/training data for reinforcement learning
    """
    start_time = dt.now(tz=TIMEZONE)

    sim = Simulator()
    query = {
        'ccy': BASKET[:1],
        'start_date': 20200809,
        'end_date': 20200810
    }
    orderbook_snapshot_history = sim.get_orderbook_snapshot_history(query=query)
    if orderbook_snapshot_history is None:
        print('Exiting: orderbook_snapshot_history is NONE')
        return

    filename = 'test_' + '{}_{}'.format(query['ccy'][0], query['start_date'])
    sim.export_to_csv(data=orderbook_snapshot_history,
                      filename=filename, compress=False)

    elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


def test_extract_features() -> None:
    """
    Test case to export *multiple* testing/training data sets for reinforcement learning
    """
    start_time = dt.now(tz=TIMEZONE)

    sim = Simulator()

    dates = [
#        20200811,
#        20200812,
        20200813,
        20200814,
        20200815,
        20200816,
        20200817,
#        20200818,
#        20200819,
#        20200820,
#        20200821,
#        20200822,
#        20200823,
#        20200824,
    ]

    
        # for ccy, ccy2 in [('LTC-USD', 'tLTCUSD')]:
    args = []

    for i, d in enumerate(dates[1:-1]):
        j = i + 2
        for ccy in EXCHANGES:
            args.append(({
                    'ccy': [ccy],  # ccy2],  # parameter must be a list
                    'start_date': dates[i],  # parameter format for dates
                    'end_date': dates[j],  # parameter format for dates
                }, 
                d,
            ))

    with parallel_backend('loky', n_jobs=4):
        Parallel()(delayed(sim.extract_features)(*arg) for arg in args)

    elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


if __name__ == '__main__':
    """
    Entry point of tests application
    """
    # test_get_tick_history()
    #test_get_orderbook_snapshot_history()
    test_extract_features()
