# Ryan Yarbrough, PhD
# this model uses historical data to generate the stock-to-flow model based
# on "Modeling Bitcoin's Value with Scarcity" from PlanB@100trillionUSD
# and project that model forward to understand where the price of btc is going

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick2_ohlc

# setting the plot back grounds to black
# plt.style.use('dark_background')

def bitcoin_monetary_inflation():
    # computing the number of btc per block
    block_reward = {}
    halving_events = 0 # block halving
    blocks_to_half = 210000 # number of blocks between halving
    max_block_reward = 50 # initial block reward
    final_block = 3150000

    for index in range(final_block):
        block = index - 1

        if block % blocks_to_half == 0 and block != 0:
            halving_events += 1

        block_btc_mined = max_block_reward/float(2**halving_events)
        block_reward.update({block:block_btc_mined})

    # computing btc produced each day
    day = []
    daily_btc = []
    height = 0
    blocks_per_day = 144
    block = 0

    while block < final_block-1:
        temp_daily_btc = []
        for _ in range(blocks_per_day):
            try:
                temp_daily_btc.append(block_reward[height])
                height += 1
                block += 1
            except KeyError:
                continue

        day.append(block)
        daily_btc.append(sum(temp_daily_btc))

    # computing monthly inflation
    inflation = np.array(daily_btc)/np.cumsum(np.array(daily_btc))

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.plot(day,np.cumsum(np.array(daily_btc)))
    plt.grid()
    plt.ylabel('Bitcoins',fontsize=12)
    ax2 = ax1.twinx()
    plt.plot(day,inflation,color='orange')
    plt.yscale('log')
    plt.ylabel('Daily Inflatoin',fontsize=12)
    plt.xlabel('Block [~10 minutes]')
    plt.title('Bitcoin Monetary Inflation',fontsize=18)
    plt.show()

def daily_btc_production(block_data,final_block):
    blocks = list(block_data['blockCount'])

    # computing the number of btc per block
    block_reward = {}
    halving_events = 0 # block halving
    blocks_to_half = 210000 # number of blocks between halving
    max_block_reward = 50 # initial block reward

    for index in range(final_block):
        block = index - 1

        if block % blocks_to_half == 0 and block != 0:
            halving_events += 1

        block_btc_mined = max_block_reward/float(2**halving_events)
        block_reward.update({block:block_btc_mined})

    # computing btc produced each day
    day = []
    daily_btc = []
    height = 0

    for index in range(len(blocks)):
        num_blocks_day = blocks[index]

        temp_daily_btc = []
        for _ in range(num_blocks_day):
            try:
                temp_daily_btc.append(block_reward[height])
                height += 1
            except KeyError:
                continue

        day.append(index)
        daily_btc.append(sum(temp_daily_btc))

    # continueing btc produced each day in ideal case
    blocks_per_day = 144
    block = sum(blocks)

    while block < final_block-1:
        temp_daily_btc = []
        for _ in range(blocks_per_day):
            try:
                temp_daily_btc.append(block_reward[height])
                height += 1
                block += 1
            except KeyError:
                continue

        day.append(index)
        daily_btc.append(sum(temp_daily_btc))

    # discounting the daily production of btc by 4% to match literature for a
    # drop of 4% each year, just apply that number to each day because that is
    # how math works
    for _ in range(np.int(np.round(len(blocks)/365))):
        daily_btc_discount = list(np.array(daily_btc)*0.96)

    return day, daily_btc_discount, daily_btc_discount

def stock_to_flow_ma(day, daily_btc, days):

    monthly_s2f = []
    market_value = []
    for index in range(days,len(day),days):
        total_btc_produced = sum(daily_btc[:index])
        monthly_btc_produced = sum(daily_btc[index-days+1:index+1])
        s2f = total_btc_produced/monthly_btc_produced
        monthly_s2f.append(s2f)

    return monthly_s2f

def price_data_interpolation(bitcoin_price, block_data):
    # finding number of days between blockchain start and exchange price
    price_start_date = bitcoin_price['date'][0]
    block_dates = list(block_data['date'])

    for index in range(len(block_dates)):
        if price_start_date == block_dates[index]:
            num_no_exchange_days = index
            break

    # building full price time series from day 0 up to last block date
    start_price = 1e-8
    start_exchange_price = bitcoin_price['USD'][0]
    interpolated_price = list(np.linspace(start_price,start_exchange_price,num_no_exchange_days))

    # adding real price data to price series
    days_to_end_block = len(block_dates[num_no_exchange_days:])
    price = interpolated_price.copy()

    price.extend(bitcoin_price['USD'])

    return price

def market_value(daily_btc, adjusted_price):
    # computing the market cap with current price and produced btc
    market_cap = []
    for index in range(1,len(adjusted_price)):
        total_btc = sum(daily_btc[:index])
        price = adjusted_price[index]
        market_cap.append(total_btc*price)

    return market_cap

def price_s2f_corrolation(adjusted_price, monthly_s2f, days):

    # shortening input data
    price = np.array(adjusted_price[0::days])
    short_s2f = np.array(monthly_s2f[:len(price)])

    # adding s2f values for known hard assets (gold,silver - from s2f paper)
    hard_asset_price = [1300,16]
    hard_asset_s2f = [62,22]

    price = np.append(price,hard_asset_price)
    short_s2f = np.append(short_s2f,hard_asset_s2f)

    # converting to log space for libear regression
    price = np.log(price)
    short_s2f = np.log(short_s2f)

    # price = np.log(np.array(adjusted_price[0::days]))
    # short_s2f = np.log(np.array(monthly_s2f[:len(price)]))

    # fitting a linear regression line to the data
    fit = np.polyfit(short_s2f[1:],price[1:],1)
    x_fit = np.linspace(0,max(short_s2f),100)

    slope = fit[0]
    intercept = fit[1]

    return slope, intercept

def projected_price(slope, intercept, s2f, market_cap, daily_btc, days):
    running_total_btc = np.cumsum(np.array(daily_btc))
    running_total_btc = np.array(running_total_btc[0::days])
    market_cap = np.array(market_cap[0::days])
    s2f = np.array(s2f)
    s2f_market_cap = np.exp(intercept)*(s2f**slope)

    # plotting stock-to-flow with price
    plt.figure(figsize=(8,6))
    plt.grid(alpha=0.5)
    plt.plot(s2f_market_cap/running_total_btc[:len(s2f_market_cap)],color='blue')
    plt.plot(market_cap/running_total_btc[:len(market_cap)],color='orange')
    plt.yscale('log')
    plt.xlabel('Months [30 days]',fontsize=12)
    plt.ylabel('Dollars [USD]',fontsize=12)
    plt.legend(['Stock-to-Flow Price','Price of BTC'])
    plt.title('Bitcoin Stock-to-Flow',fontsize=18)
    plt.show()

def halving_candles(market_cap, daily_btc):
    # computing number of days for a halving
    ideal_block_time = 10 # minutes
    blocks_to_half = 210000 # number of blocks between halving
    days_between_halving = int(blocks_to_half*ideal_block_time/(60*24))
    number_halving = int(len(market_cap)/days_between_halving)
    running_total_btc = np.cumsum(np.array(daily_btc))
    btc_price_day = market_cap/running_total_btc[:len(market_cap)]

    # lists of price data for a candle chart
    o = []
    c = []
    h = []
    l = []

    # getting candle data for each halving
    ini_day = 0
    for _ in range(number_halving):
        end_day = ini_day + days_between_halving

        o.append(btc_price_day[ini_day])
        c.append(btc_price_day[end_day])
        h.append(max(btc_price_day[ini_day:end_day]))
        l.append(min(btc_price_day[ini_day:end_day]))

        ini_day = end_day

    # getting current halving candle data (this candle is not complete)
    o.append(btc_price_day[ini_day])
    c.append(btc_price_day[-1])
    h.append(max(btc_price_day[ini_day:]))
    l.append(min(btc_price_day[ini_day:]))

    plt.figure(figsize=(8,6))
    plt.grid(alpha=0.5)
    candlestick2_ohlc(plt.gca(),o,h,l,c,colorup='green',colordown='red',width=0.75)
    plt.yscale('log')
    plt.xlim([-0.5,3.5])
    plt.xticks([0,1,2,3],['Cycle 1', 'Cycle 2', 'Cycle 3', 'Cycle 4'])
    plt.xlabel('Bitcoin Halving Cycles [~4 years]',fontsize=12)
    plt.ylabel('Dollars [USD]',fontsize=12)
    plt.title('Bitcoin Preformance Every Halving Cycle',fontsize=18)
    plt.show()

# simulation constants
final_block = 2000000
days = 30

# importing data
bitcoin_price = './bitcoin_price.csv'
bitcoin_price = pd.read_csv(bitcoin_price)
block_data = './bitcoin_block_data_short.csv'
block_data = pd.read_csv(block_data)

adjusted_price = price_data_interpolation(bitcoin_price, block_data)
day, daily_btc, daily_btc_discount = daily_btc_production(block_data, final_block)
market_cap = market_value(daily_btc_discount, adjusted_price)
s2f = stock_to_flow_ma(day, daily_btc, days)
slope, intercept = price_s2f_corrolation(market_cap, s2f, days)



# running_total_btc = np.cumsum(np.array(daily_btc))
# running_total_btc = np.array(running_total_btc)
# market_cap = np.array(market_cap)
# s2f = np.array(s2f)
# s2f_market_cap = np.exp(intercept)*(s2f**slope)
# s2f_price = s2f_market_cap/running_total_btc[:len(s2f_market_cap)]
#
# print(len(daily_btc[0::30]))
# print(len(s2f_price))
#
#
#
#
# plt.plot(s2f_price*daily_btc[30::30]/30)
# plt.yscale('log')
#
# # plt.figure()
# # plt.plot(daily_btc)
# plt.show()
#
# exit()



# bitcoin_monetary_inflation()
# halving_candles(market_cap, daily_btc)
projected_price(slope, intercept, s2f, market_cap, daily_btc, days)
