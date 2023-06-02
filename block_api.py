import json
import numpy as np
from pprint import pprint
from datetime import datetime
from urllib.request import urlopen

# this comes from the price data
max_block = 688244

data = {}

block = 666673
last_date = None
while block < max_block:
    # getting the block hash
    url = 'https://mempool.space/api/block-height/' + str(block)
    hash = urlopen(url).read().decode("utf-8")
    print('block hash: ' + str(hash))

    url = 'https://mempool.space/api/block/' + str(hash)
    json_url = urlopen(url)
    json_url = json.loads(json_url.read())

    # finding the block date
    timestamp = json_url['timestamp']
    date = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')

    # seeing if a new day has started
    if date not in data.keys():
        data.update({date:[]})

        # seeing if data can be added to the data file
        if last_date != None:
            total_blocks = len(data[last_date])
            file = open('block_dates.csv','a')
            file.write(date + ',' + str(total_blocks) + '\n')
            file.close()

            print(str(total_blocks) + ' blocks on ' + str(date) + ', block height: ' + str(block))

        # itterating date
        last_date = date

    # adding the block number to the day
    data[date].append(block)

    # itterating the block
    block += 1
