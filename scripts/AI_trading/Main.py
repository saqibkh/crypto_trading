


import os
import sys
import time
import datetime


import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ---------Set sys.path for MAIN execution---------------------------------------
full_path = os.path.abspath(os.path.dirname(sys.argv[0])).split('crypto_trading')[0]
full_path += "crypto_trading"
sys.path.append(full_path)
# Walk path and append to sys.path
for root, dirs, files in os.walk(full_path):
    for dir in dirs:
        sys.path.append(os.path.join(root, dir))

import ALPACA
import constants
import sim_logging

class Key:
    def __init__(self, i_key, i_secret_key, i_url):
        self.key = i_key
        self.secretKey = i_secret_key
        self.url = i_url

def usage():
    print("Usage: Provide a path to the key file.")
    print("Example: python3 scripts/AI_trading/Main.py -key /home/saqib/.ssh/alpaca_paper_keys")

def main(argv):
    i_keys = None

    for i in range(len(argv)):
        if argv[i] == '-key':
            i_alpaca_key_file = argv[i+1]
            with open(i_alpaca_key_file, 'r') as file:
                for line in file:
                    if line.startswith('Key:'):
                        i_key = line.split(':')
                    elif line.startswith('Secret_Key:'):
                        i_secretkey = line.split(':')
                    elif line.startswith('URL:'):
                        i_url = line.split(':', 1)
            i_keys = Key(i_key, i_secretkey, i_url)

    # It is required to provide key. Otherwise we can't buy/sell
    if i_keys is None:
        usage()
        exit(-1)
    

    i_log_directory = os.path.abspath(os.path.dirname(sys.argv[0])).split('crypto_trading')[0]
    i_log_directory += "crypto_trading/logs/"
    simlog = sim_logging.SIMLOG(log_dir=i_log_directory)

    # Step-1: Get data for all crypto currencies
    i_alpaca_object = ALPACA.ALPACA(i_keys, simlog)
    i_crypto_list = i_alpaca_object.get_all_crypto_currencies()

    for crypto in i_crypto_list:
        # We only care about crypto value in USD. Thus ignore all USDT and BTC conversion coins
        if crypto.endswith('/USD'):
            simlog.info("==================================================================================")
            simlog.info("Starting to process crypto: " + str(crypto))
            crypto = crypto.replace('/', '-')
            # Step-2.0 If latest AI models doesn't exist for these coins, then start pulling historical data
              # Step-2.1: Create AI models and use that to calculate RMSE/sharpeRatio/nextDayPrice
              # Step-2.2: Using AI models and current price, determine if it is a BUY/SELL
            i_action = i_alpaca_object.crypto_prediction(crypto)

            # Step-3: If a BUY/SELL execute that trade
            if i_action != constants.CRYPTO_LEAVE:
                i_alpaca_object.process_action(crypto, i_action)

            simlog.info("Pause for 1 second")
            time.sleep(1)
            simlog.info("==================================================================================")

if __name__ == "__main__":
    main(sys.argv[1:])
