class SlippagePct:
    def __init__(self, fixed_percent = 0.001):
        self.fixed_percent = fixed_percent

    def getRealBuyPrice(self, plan_price):
        return plan_price * (1 + self.fixed_percent)

    def getRealSellPrice(self, plan_price):
        return plan_price * (1 - self.fixed_percent)

    def setPercent(self, percent):
        self.fixed_percent = percent

class SlippagePrc:
    def __init__(self, fixed_price = 0.01):
        self.fixed_price = fixed_price

    def getRealBuyPrice(self, plan_price):
        return plan_price + self.fixed_price

    def getRealSellPrice(self, plan_price):
        return plan_price - self.fixed_price

    def setFixedPrice(self, price):
        self.fixed_price = price

if __name__ == "__main__":
    sliper = SlippagePct()
    print(sliper.getRealBuyPrice(1000))
    print(sliper.getRealSellPrice(1000))
    sliper = SlippagePrc()
    print(sliper.getRealBuyPrice(1000))
    print(sliper.getRealSellPrice(1000))

