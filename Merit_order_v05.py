# %% Import relevant packages
import pandas as pd
import matplotlib.pyplot as plt

# %% Calculate the marginal cost of a power plant
def calculate_marginal_cost(pp_dict, fuel_prices, emission_factors):
    """
    Calculates the marginal cost of a power plant based on the fuel costs and efficiencies of the power plant.

    Parameters
    ----------
    pp_dict : dict
        Dictionary of power plant data.
    fuel_prices : dict
        Dictionary of fuel data.
    emission_factors : dict
        Dictionary of emission factors.  

    Returns
    -------
    marginal_cost : float
        Marginal cost of the power plant.

    """

    fuel_price = fuel_prices[pp_dict['technology']]
    emission_factor = emission_factors['emissions'].at[pp_dict['technology']]
    co2_price = fuel_prices['co2']

    fuel_cost = fuel_price/pp_dict['efficiency']
    emissions_cost = co2_price*emission_factor/pp_dict['efficiency']
    variable_cost = pp_dict['var_cost']
    
    marginal_cost = fuel_cost + emissions_cost + variable_cost

    return marginal_cost

# %% Calculate the market clearing price
def calculate_market_clearing_price(powerplants, demand, feed_in):
    """
    Calculates the market clearing price of the merit order model.

    Parameters
    ----------
    powerplants : pandas.DataFrame
        Dataframe containing the power plant data.
    demand : float
        Demand of the system.

    Returns
    -------
    mcp : float
        Market clearing price.

    """
    
    # Sort the power plants on marginal cost
    powerplants.sort_values('marginal_cost', inplace=True)
    
    # Calculate the cumulative capacity
    powerplants['cumulative_capacity'] = powerplants.capacity.cumsum() + feed_in
    
    # Calculate the market clearing price
    if powerplants['cumulative_capacity'].iat[-1] < demand:
        mcp = powerplants['marginal_cost'].iat[-1]
    else:
        mcp = powerplants.loc[powerplants['cumulative_capacity'] >= demand, 'marginal_cost'].iat[0]
    
    return mcp

# %% Plot merit order curve
def assign_color(powerplant):
    """
    Assigns a color to a power plant based on the technology.

    Parameters
    ----------
    powerplant : pandas.Series
        Series containing the power plant data.

    Returns
    -------
    color : str
        Color of the power plant.

    """
    
    technology = powerplant['technology']
    color = colors[technology]
    
    return color


def plot_merit_order_curve(powerplants, mcp, demand, feed_in):
    """
    Plots the merit order curve.

    Parameters
    ----------
    powerplants : pandas.DataFrame
        Dataframe containing the power plant data.
    demand : float
        Demand of the system.

    Returns
    -------
    None.

    """
    
    # Plot the merit order curve
    pp = powerplants.sort_values('marginal_cost')

    plt.plot()
    plt.bar(x=0, height=1, width=feed_in, color='blue', label='Renewables', align='edge', alpha=0.4, edgecolor='k')

    plt.bar(x=feed_in+pp.capacity.cumsum() - pp.capacity,
            height=pp['marginal_cost'],
            width=pp.capacity,
            align='edge',
            color=pp['color'],
            alpha=0.4,
            edgecolor='k')

    plt.xlim(left=0)
    plt.plot([demand,demand], [0,mcp], 'r--', label='demand')
    plt.plot([0,demand], [mcp,mcp], 'r--', label='mcp')
    plt.xlabel('Marginal cost')
    plt.ylabel('Capacity')
    plt.title('Merit order curve')
    plt.show()

# %% Define the required dictionaries
powerplants = pd.read_csv('inputs_2/2020_majorPowerplants_GER_1h.csv', index_col=0)
fuel_prices = pd.read_csv('inputs_2/2020_fuelPrices_GER_1h.csv', index_col=0, parse_dates=True)
emission_factors = pd.read_csv('inputs_2/2020_emissionFactors_GER_1h.csv', index_col=0)

demand_df = pd.read_csv('inputs_2/2020_demand_GER_1h.csv', index_col=0, parse_dates=True)
feed_in_df = pd.read_csv('inputs_2/2020_renewablesCF_GER_1h.csv', index_col=0, parse_dates=True)

# Installed renewable Capacity in MW
installed_pv = 48206
installed_onshore_wind = 53184
installed_offshore_wind = 7504

feed_in_df['solar'] *= installed_pv
feed_in_df['onshore'] *= installed_onshore_wind
feed_in_df['offshore'] *= installed_offshore_wind

# %%
colors = {'nuclear': 'green',
          'lignite': 'brown',
          'hard coal': 'black',
          'natural gas': 'red',
          'oil': 'yellow'
          }

#Calculate the marginal cost of each power plant
marginal_costs = powerplants.apply(calculate_marginal_cost, axis=1, fuel_prices=fuel_prices, emission_factors=emission_factors).T

# %%
mcp_df = pd.DataFrame(columns=['mcp'], index=demand_df.index, data=0.)
for i in range(len(demand_df)):
    pp_df = powerplants.copy()
    pp_df['marginal_cost'] = marginal_costs.iloc[i]
    mcp = calculate_market_clearing_price(pp_df,
                                          demand_df['demand'].iat[i],
                                          feed_in_df.iloc[i].sum())
    mcp_df['mcp'].iat[i] = mcp

# %%
powerplants['color'] = powerplants.apply(assign_color, axis=1)

# we now adjuts the parameters sent to the plot function to pass only single values at a given timestep
# when working with datetime indexes, we can use a date as index. 
# But keep in mind that we need to change iloc and iat to loc and at

timestep = '2020-01-01 00:00:00'
pp_df = powerplants.copy()
pp_df['marginal_cost'] = marginal_costs.loc[timestep]

plot_merit_order_curve(pp_df,
                       mcp = mcp_df['mcp'].at[timestep],
                       demand = demand_df['demand'].at[timestep],
                       feed_in = feed_in_df.loc[timestep].sum())

# %%


