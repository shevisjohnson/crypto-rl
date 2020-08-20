# meta_broker.py
#
#   Wrapper class implementing broker.py to manage/monitor a portfolio of
#   multiple currencies and the exchanges between them.
#
#   e.g. 
#       currencies: ['USD', 'BTC', 'ETH']
#       exchanges: ['BTC-USD', 'ETH-BTC', 'ETH-USD']
#

from configurations import LOGGER