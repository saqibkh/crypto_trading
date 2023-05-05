#!/usr/bin/python


##############################################################################
#
# This file is used to pull crypto market data from the web and into the
# log directory.
#
##############################################################################

import sys
import logging
import time
import string
import datetime
import getopt
import random
import subprocess
import os
import csv
import yfinance as yf

# ---------Set sys.path for MAIN execution---------------------------------------
full_path = os.path.abspath(os.path.dirname(sys.argv[0])).split('crypto_trading')[0]
full_path += "crypto_trading"
sys.path.append(full_path)
# Walk path and append to sys.path
for root, dirs, files in os.walk(full_path):
    for dir in dirs:
        sys.path.append(os.path.join(root, dir))

import constants

def main(argv):
    i_base_directory = os.path.abspath(os.path.dirname(sys.argv[0])).split('crypto_trading')[0]
    i_log_directory = i_base_directory + "crypto_trading" + "/" + "logs" + "/"

    # By Default get the short list of stock names
    i_crypto_list = constants.crypto_list

    print("Start pulling in stock prices")
    for i in range(len(i_crypto_list)):
        try:
            data = yf.download(tickers=i_crypto_list[i], period='120mo', interval='1d')
            data.to_csv(i_log_directory + i_crypto_list[i] + '.csv')
        except Exception as e:
            print("Some error has occured. Continue!!!!!")
            print(e)

    print("Finished get all the required stock logs")

if __name__ == "__main__":
    main(sys.argv[1:])
