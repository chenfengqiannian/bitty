import logging
import time
import sys
import csv

import os
import types

from btfxwss import BtfxWss
import logging
import json
import time
import ssl
import websocket
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
if __name__=="__main__":
    import sys
    coin_type=sys.argv[1]
    def _connect(self):
        """Creates a websocket connection.

        :return:
        """
        self.log.debug("_connect(): Initializing Connection..New")
        self.socket = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )

        if 'ca_certs' not in self.sslopt.keys():
            ssl_defaults = ssl.get_default_verify_paths()
            self.sslopt['ca_certs'] = ssl_defaults.cafile

        self.log.debug("_connect(): Starting Connection..")
        self.socket.run_forever(sslopt=self.sslopt)

        while self.reconnect_required.is_set():
            if not self.disconnect_called.is_set():
                self.log.info("Attempting to connect again in %s seconds.reconnect gao"
                              % self.reconnect_interval)
                self.state = "unavailable"
                time.sleep(self.reconnect_interval)
                self.log.debug("_connect():  ReConnection..")
                self.socket = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )

                # We need to set this flag since closing the socket will
                # set it to False
                self.socket.keep_running = True
                self.socket.run_forever(sslopt=self.sslopt)
    wss = BtfxWss()
    wss.conn._connect= types.MethodType(_connect, wss.conn)
    wss.start()



    while not wss.conn.connected.is_set():
        time.sleep(1)

    # Subscribe to some channels
    wss.subscribe_to_ticker(coin_type)
    # wss.subscribe_to_order_book('BTCUSD')

    # Do something else
    t = time.time()
    while time.time() - t < 10:
        pass

    # Accessing data stored in BtfxWss:
    ticker_q = wss.tickers(coin_type)  # returns a Queue object for the pair.


    with open(coin_type+".csv", "a+") as csvfile:
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
    wss.unsubscribe_from_ticker(coin_type)
    # wss.unsubscribe_from_order_book('BTCUSD')

    # Shutting down the client:
    wss.stop()
