import logging
import time
import sys
import csv

import os
from btfxwss import BtfxWss

log = logging.getLogger(__name__)

fh = logging.FileHandler('test.log')
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)

log.addHandler(sh)
log.addHandler(fh)
logging.basicConfig(level=logging.DEBUG, handlers=[fh, sh])
import signal
# Define signal handler function

wss = BtfxWss()
wss.start()

while not wss.conn.connected.is_set():
    time.sleep(1)

# Subscribe to some channels
wss.subscribe_to_ticker('ETHUSD')
# wss.subscribe_to_order_book('BTCUSD')

# Do something else
t = time.time()
while time.time() - t < 10:
    pass

# Accessing data stored in BtfxWss:
ticker_q = wss.tickers('ETHUSD')  # returns a Queue object for the pair.


with open("ethusdtbitfinex.csv", "a+") as csvfile:
    while True:
        date=ticker_q.get()
        datelist=list()
        datelist.append(str(date[1]))
        datelist.append(str(date[0][0][0]))
        datelist.append(str(date[0][0][7]))
        csvfile.write(",".join(datelist)+"\n")
        csvfile.flush()
        print(datelist)




# Unsubscribing from channels:
wss.unsubscribe_from_ticker('ETHUSD')
# wss.unsubscribe_from_order_book('BTCUSD')

# Shutting down the client:
wss.stop()
