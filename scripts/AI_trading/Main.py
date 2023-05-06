


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

def main(argv):
    i_log_directory = os.path.abspath(os.path.dirname(sys.argv[0])).split('crypto_trading')[0]
    i_log_directory += "crypto_trading/logs/"
    simlog = sim_logging.SIMLOG(log_dir=i_log_directory)

    # Step-1: Get data for all crypto currencies
    i_alpaca_object = ALPACA.ALPACA(simlog)
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
