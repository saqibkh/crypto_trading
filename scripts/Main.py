#!/usr/bin/python

# This script will be used to alert data from api-whales-alert.io.
# We will use that information to get more data from explorer blockchain.com
# to determine if a whale bought or sold certain crypto currencies.
# Then act accordingly

import os
import sys
import csv


# ---------Set sys.path for MAIN execution---------------------------------------
full_path = os.path.abspath(os.path.dirname(sys.argv[0])).split('crypto_trading')[0]
full_path += "crypto_trading"
sys.path.append(full_path)
# Walk path and append to sys.path
for root, dirs, files in os.walk(full_path):
    for dir in dirs:
        sys.path.append(os.path.join(root, dir))

import whale_io
import constants
import blockchain

def main(argv):

    # Step-1: Get whales transaction data from https://api.whales-alert-io
    i_data = whale_io.Whale_Alert()

    # Step-2:
    l_result = blockchain.Blockchain()
    x = 1


if __name__ == "__main__":
    main(sys.argv[1:])

