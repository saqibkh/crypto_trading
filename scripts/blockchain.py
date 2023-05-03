#!/usr/bin/python

import time
import requests
import json


class Blockchain:
    def __init__(self):
        api_key = 'f1093139-edc3-4c44-9375-b91cb767c10e'
        crypto = 'ethereum'  # cryptocurrency symbol
        address = '72b2c54c7f83fa9cdb023131a95d96b1adec3126'

        # Define the API endpoint URL and parameters
        url = f'https://blockchain.info/rawaddr/${address}'

        # Make a GET request to the API endpoint
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Extract relevant data from the response JSON
            data = response.json()['data'][address]['address']
            balance = data['balance']
            total_received = data['total_received']
            total_sent = data['total_spent']
            transaction_count = data['transaction_count']
            # Print out the extracted data
            print(f"Balance: {balance}")
            print(f"Total Received: {total_received}")
            print(f"Total Sent: {total_sent}")
            print(f"Transaction Count: {transaction_count}")
        else:
            print("Error: Request failed")