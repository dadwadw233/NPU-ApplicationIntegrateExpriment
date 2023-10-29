import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import time

timestamps = []
prices = []

refresh_interval = 1

plt.figure(figsize=(12, 6))

while True:
    try:
        response = requests.get("https://api.coindesk.com/v1/bpi/currentprice.json")
        data = response.json()
        price_in_usd = float(data["bpi"]["USD"]["rate"].replace(",", ""))
        timestamp = datetime.now()

        timestamps.append(timestamp)
        prices.append(price_in_usd)

        if len(prices) >= 2:
            price_change = prices[-1] - prices[-2]
            if price_change > 0:
                line_color = 'g'
            elif price_change < 0:
                line_color = 'r'
            else:
                line_color = 'b'

            plt.clf()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.plot(timestamps, prices, marker='o', color=line_color, label='BTC')
            plt.gcf().autofmt_xdate()
            plt.title('Bitcoin Price Trend')
            plt.xlabel('Time')
            plt.ylabel('Price (USD)')
            plt.grid(True)

            plt.annotate(f'{price_in_usd:.2f} USD', (timestamp, price_in_usd), textcoords="offset points",
                         xytext=(0, 10), ha='center')

            plt.legend()
            plt.tight_layout()
            plt.draw()
            plt.pause(refresh_interval)

    except Exception as e:
        print(f"发生错误：{e}")

    time.sleep(refresh_interval)


