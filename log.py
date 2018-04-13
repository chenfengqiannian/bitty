import logging

import sys

log = logging.getLogger(__name__)
fh = logging.FileHandler('test.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[sh, fh])
