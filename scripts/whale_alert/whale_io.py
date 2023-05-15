#!/usr/bin/python

import os
import sys
import csv
import time
import calendar
import requests
import json
import constants
from datetime import datetime


apikey='EVfmixRGs9j0a69jYFfcGNj058yPAmIV'



class TRANSACTION:
    def __init__(self, i_transaction_data):

        self.raw_data = i_transaction_data
        self.blockchain = i_transaction_data['blockchain']
        self.transaction_id = i_transaction_data['id']
        self.transaction_type = i_transaction_data['transaction_type']
        self.transaction_hash = i_transaction_data['hash']

        self.source_address = i_transaction_data['from']['address']
        self.source_owner = i_transaction_data['from']['owner']
        self.source_owner_type = i_transaction_data['from']['owner_type']

        self.destination_address = i_transaction_data['to']['address']
        self.destination_owner = i_transaction_data['to']['owner']
        self.destination_owner_type = i_transaction_data['to']['owner_type']

        self.timestamp = i_transaction_data['timestamp']
        self.amount = i_transaction_data['amount']
        self.amount_usd = i_transaction_data['amount_usd']

class Whale_Alert():
    def __init__(self):

        # Create an empty array to fill with transactions later on
        self.transactions = []

        self.whale_io_apikey = 'EVfmixRGs9j0a69jYFfcGNj058yPAmIV'
        now = datetime.utcnow()
        unixtime = calendar.timegm(now.utctimetuple())
        self.end_time = unixtime
        self.start_time = str(self.end_time - 60 * 60)  # retrieve data for last 60 minutes

        self.min_values = str(1000000)  # Above 1 million dollars

        url = 'https://api.whale-alert.io/v1/transactions?api_key=' + self.whale_io_apikey + '&min_value=' + self.min_values + "&start=" + self.start_time

        # Send a GET request to the Whale Alert API with the specified parameters
        self.whale_data_raw = requests.get(url).json()

        # Extract just the transaction data.
        try:
            transactions = self.whale_data_raw['transactions']
        except Exception as e:
            # Sometimes it gives an error for God knows what reason.
            # So just retry and it usually works
            time.sleep(4)
            transactions = self.whale_data_raw['transactions']

        for transaction in transactions:
            l_result = self.add_to_master_list(transaction)
            if l_result == constants.PASS:
                self.transactions.append(TRANSACTION(transaction))

    def add_to_master_list(self, transaction):

        # This list holds all the transaction ids in the masterlist, so as not to repeat the same entry twice
        i_masterlist_ids = []

        log_master_filename = "MasterList.csv"
        logfile_path = os.path.abspath(os.path.dirname(sys.argv[0])).split('crypto_trading')[0] + \
                   "crypto_trading/logs/" + log_master_filename

        # Create a new file if the file doesn't already exists
        if os.path.isfile(logfile_path):
            #print("Master file already exists. Append to it")
            i_masterlist_ids = self.read_master_list_ids(logfile_path, 'id')

        else:
            # If the file doesn't exist, create it and append to it
            with open(logfile_path, 'a') as file:
                file.write("date,blockchain,symbol,id,transaction_type,hash,from_address,from_owner,from_owner_type,to_address,to_owner,to_owner_type,timestamp,amount,amount_usd\n")

        if transaction['id'] not in i_masterlist_ids:
            with open(logfile_path, 'a') as file:
                i_timestamp = int(transaction['timestamp'])
                # Convert the timestamp to a datetime object
                date_string = str(datetime.fromtimestamp(i_timestamp)) #2023-05-15 10:58:00

                file.write(str(date_string)+',')
                file.write(str(transaction['blockchain'])+',')
                file.write(str(transaction['symbol']) + ',')
                file.write(str(transaction['id']) + ',')
                file.write(str(transaction['transaction_type']) + ',')
                file.write(str(transaction['hash']) + ',')
                file.write(str(transaction['from']['address']) + ',')
                file.write(str(transaction['from']['owner']) + ',')
                file.write(str(transaction['from']['owner_type']) + ',')
                file.write(str(transaction['to']['address']) + ',')
                file.write(str(transaction['to']['owner']) + ',')
                file.write(str(transaction['to']['owner_type']) + ',')
                file.write(str(transaction['timestamp']) + ',')
                file.write(str(transaction['amount']) + ',')
                file.write(str(transaction['amount_usd']) + '\n')
            return constants.PASS
        else:
            return constants.SKIP

    def read_master_list_ids(self, i_file, i_column_name):
        with open(i_file, 'r') as file:
            reader = csv.DictReader(file)
            column_data = []
            for row in reader:
                column_data.append(row[i_column_name])
        return column_data
