theory_txt = """
    ### Portfolio Theory:
    
    Imagine you have some money to invest, and you're considering buying different things with it, like stocks, bonds, or maybe even real estate. Portfolio theory is a way to help you decide how to mix these investments together to get the best possible return while managing risk.
    
    The basic idea is that by spreading your money across different types of investments, you can reduce the overall risk of your portfolio. This is because if one investment does poorly, hopefully, another one will do well enough to make up for it.
    
    But it's not just about spreading your money around randomly. Portfolio theory uses mathematical formulas to help you find the best mix of investments based on things like:
    
    - **Expected return:** How much money you expect to make from each investment.
    - **Risk:** How much the value of each investment might go up and down over time.
    - **Correlation:** How closely the returns of different investments are related to each other.
    
    By considering these factors, portfolio theory helps you build a portfolio that gives you the best possible return for the amount of risk you're willing to take.
    
    ### Optimization:
    
    Once you've decided on the types of investments you want to include in your portfolio, optimization is about finding the best way to allocate your money among them. It's like solving a puzzle to figure out the right balance that maximizes your return while minimizing your risk.
    
    Optimization algorithms use mathematical techniques to explore different combinations of investments and find the one that gives you the best results according to your goals and constraints. For example, you might want to maximize your return while keeping your risk below a certain level, or you might want to minimize your risk while aiming for a specific level of return.
    
    ### Rolling Windows for Backtesting:
    
    Backtesting is a way to test how well a trading strategy or investment approach would have performed in the past. It's like simulating your strategy in historical market conditions to see how it would have fared.
    
    One challenge in backtesting is deciding how much historical data to use. Using too much data may not accurately reflect how your strategy would perform in current market conditions, while using too little data may not provide a robust assessment of its effectiveness.
    
    This is where rolling windows come in. Instead of testing your strategy on the entire historical dataset at once, rolling windows divide the data into smaller, overlapping chunks, and test the strategy on each chunk separately. For example, if you have 10 years of historical data, you might use rolling windows of 1-year periods, moving forward one year at a time.
    
    By using rolling windows, you can evaluate the performance of your portfolio strategy across different periods of time, capturing various market conditions such as bull markets, bear markets, and periods of volatility. This approach helps to reduce the uncertainty associated with selecting the amount of historical data to use, as it provides a more comprehensive view of your strategy's performance over time.
    """