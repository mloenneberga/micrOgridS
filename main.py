import logging
import custom_constraints as constraints
import pandas as pd
import os
import time

from oemof.outputlib import processing, views
from oemof.solph import (Sink, Transformer, Source, Bus, Flow, NonConvex,
                         Model, EnergySystem, components, custom, Investment)
from oemof.tools import logger, economics
from oemof.network import Node
import cost_summary as lcoe

# cost dictionary #####################################################################################################
#######################################################################################################################

def get_cost_dict(PH):
    cost = {'pp_oil_1': {'fix': economics.annuity( (500 / 8760) * PH, 20, 0.094 ),
                         'var': 1.2,
                         'o&m': 0.02},
            'pp_oil_2': {'fix': economics.annuity( (500 / 8760) * PH, 20, 0.094 ),
                         'var': 1.2,
                         'o&m': 0.02},
            'pp_oil_3': {'fix': economics.annuity( (500 / 8760) * PH, 20, 0.094 ),
                         'var': 1.2,
                         'o&m': 0.02},
            'storage': {'fix': (3.88 / 8760) * PH,
                        'var': 0.087,
                        'epc': economics.annuity( (300 / 8760) * PH, 10, 0.094 )},
            'pv': {'fix': (25 / 8760) * PH,
                   'var': 0,
                   'epc': economics.annuity( (2500 / 8760) * PH, 20, 0.094 )}
            }
    return cost


def get_sim_params(cost):
    sim_params = {'pv': {'nominal_capacity': 265.017017,
                         'investment': Investment( ep_costs=cost['pv']['epc'] )},
                  'storage': {'nominal_capacity': 268.1211092,
                              'investment': Investment( ep_costs=cost['storage']['epc'] )}}
    return sim_params


def add_inverter(i, o, name, eta=1):
    return Transformer( label=name,
                        inputs={i: Flow()},
                        outputs={o: Flow()},
                        conversion_factors={o: eta} )


def diesel_only(mode, feedin, initial_batt_cap, cost, iterstatus=None, PV_source=True, storage_source=True,logger=False):

    if logger==1:
        logger.define_logging()

    ##################################### Initialize the energy system##################################################

    # times = pd.DatetimeIndex(start='04/01/2017', periods=10, freq='H')
    times = feedin.index

    energysystem = EnergySystem( timeindex=times )

    # switch on automatic registration of entities of EnergySystem-object=energysystem

    Node.registry = energysystem

    # add components

    b_el = Bus( label='electricity' )
    b_dc = Bus( label='electricity_dc' )
    b_oil = Bus( label='diesel_source' )

    demand_feedin = feedin['demand_el']

    Sink( label='demand',
          inputs={b_el: Flow( actual_value=demand_feedin,
                              nominal_value=1,
                              fixed=True )} )

    Sink( label='excess',
          inputs={b_el: Flow()} )


    Source( label='diesel',
            outputs={b_oil: Flow()} )

    generator1 = custom.DieselGenerator( label='pp_oil_1',
                                         fuel_input={b_oil: Flow( variable_costs=cost['pp_oil_1']['var'] )},
                                         electrical_output={b_el: Flow( nominal_value=186,
                                                                        min=0.3,
                                                                        max=1,
                                                                        nonconvex=NonConvex(
                                                                            om_costs=cost['pp_oil_1']['o&m'] ),
                                                                        fixed_costs=cost['pp_oil_1']['fix']
                                                                        )},
                                         fuel_curve={'1': 42, '0.75': 33, '0.5': 22, '0.25': 16} )

    generator2 = custom.DieselGenerator( label='pp_oil_2',
                                         fuel_input={b_oil: Flow( variable_costs=cost['pp_oil_2']['var'] )},
                                         electrical_output={b_el: Flow( nominal_value=186,
                                                                        min=0.3,
                                                                        max=1,
                                                                        nonconvex=NonConvex(
                                                                            om_costs=cost['pp_oil_2']['o&m'] ),
                                                                        fixed_costs=cost['pp_oil_2']['fix'],
                                                                        variable_costs=0 )},
                                         fuel_curve={'1': 42, '0.75': 33, '0.5': 22, '0.25': 16} )

    generator3 = custom.DieselGenerator( label='pp_oil_3',
                                         fuel_input={b_oil: Flow( variable_costs=cost['pp_oil_3']['var'] )},
                                         electrical_output={b_el: Flow( nominal_value=320,
                                                                        min=0.3,
                                                                        max=1,
                                                                        nonconvex=NonConvex(
                                                                            om_costs=cost['pp_oil_3']['o&m'] ),
                                                                        fixed_costs=cost['pp_oil_3']['fix'],
                                                                        variable_costs=0 )},
                                         fuel_curve={'1': 73, '0.75': 57, '0.5': 38, '0.25': 27} )

    # List all generators in a list called gen_set
    gen_set = [generator1, generator2, generator3]

    sim_params = get_sim_params(cost)

    if mode == 'simulation':
        nominal_cap_pv = sim_params['pv']['nominal_capacity']
        inv_pv = None
        nominal_cap_batt = sim_params['storage']['nominal_capacity']
        inv_batt = None
    elif mode == 'investment':
        nominal_cap_pv = None
        inv_pv = sim_params['pv']['investment']
        nominal_cap_batt = None
        inv_batt = sim_params['storage']['investment']
    else:
        raise (UserWarning, 'Energysystem cant be build. Check if mode is spelled correctely. '
                            'It can be either [simulation] or [investment]')

    if PV_source == 1:
        PV = Source( label='PV',
                     outputs={b_dc: Flow( nominal_value=nominal_cap_pv,
                                          fixed_costs=cost['pv']['fix'],
                                          actual_value=feedin['PV'],
                                          fixed=True,
                                          investment=inv_pv )} )
    else:
        PV=None

    if storage_source == 1:
        storage = components.GenericStorage( label='storage',
                                             inputs={b_dc: Flow()},
                                             outputs={b_dc: Flow( variable_costs=cost['storage']['var'],
                                                                  fixed_costs=cost['storage']['fix'])},
                                             nominal_capacity=nominal_cap_batt,
                                             capacity_loss=0.00,
                                             initial_capacity=initial_batt_cap,
                                             nominal_input_capacity_ratio=0.546,
                                             nominal_output_capacity_ratio=0.546,
                                             inflow_conversion_factor=0.92,
                                             outflow_conversion_factor=0.92,
                                             capacity_min=0.5,
                                             capacity_max=1,
                                             investment=inv_batt,
                                             initial_iteration=iterstatus )
    else:
        storage=None

    if storage_source == 1 or PV_source == 1:
        inverter1 = add_inverter( b_dc, b_el, 'Inv_pv' )

    ################################# optimization ############################
    # create Optimization model based on energy_system
    logging.info( "Create optimization problem" )

    m = Model( energysystem )

    ################################# constraints ############################

    sr_requirement = 0.2
    sr_limit = demand_feedin * sr_requirement

    rm_requirement = 0.4
    rm_limit = demand_feedin * rm_requirement

    constraints.spinning_reserve_constraint( m, sr_limit, groups=gen_set, storage=storage )

    # constraints.n1_constraint(m, demand_feedin, groups=gen_set)

    constraints.gen_order_constraint( m, groups=gen_set )

    constraints.rotating_mass_constraint( m, rm_limit, groups=gen_set, storage=storage )

    return [m, gen_set]


def solve_and_create_results(m, lp_write=False, gap=0.01):
    if lp_write == True:
        m.write( os.path.join( 'results', 'Lifuka.lp' ), io_options={'symbolic_solver_labels': True} )

    # solve with specific optimization options (passed to pyomo)
    logging.info( "Solve optimization problem" )

    m.solve( solver='gurobi', solve_kwargs={'tee': True}, cmdline_options={'MIPGap': gap} )

    # cmdline_options = {'MIPGap': 0.01}

    # write back results from optimization object to energysystem
    logging.info( 'Print results back to energysystem' )
    res = processing.results( m )

    # ToDo work on processing of the results
    # ToDo add constraints to another module, generalize demand

    return res


def sizing_results(results,m, sizing_list):
    res = {}

    for i in range( len( sizing_list ) ):
        node=m.es.groups[sizing_list[i]]
        node_data=views.node(results,node)
        for nodes, flow_name in node_data['sequences']:
            if 'scalars' in node_data and node_data['scalars'].get((nodes, 'invest')) is not None:
                res[nodes] = node_data['scalars'].get((nodes, 'invest'))

        df=pd.DataFrame.from_dict(res,orient='index')
    return df


def results_postprocessing(n, component_list, time_horizon=None):
    generator_list = []

    for i in range( len( component_list ) ):
        d1 = views.node( n, component_list[i] )['sequences']

        if time_horizon is not None:
            generator_list.append( d1.iloc[:time_horizon] )
        else:
            generator_list.append( d1 )

    res = pd.concat( generator_list, axis=1 )

    return res


def get_timeseries(file='data/timeseries.csv'):
    timeseries = pd.read_csv(file, sep=';' )
    timeseries.set_index( pd.DatetimeIndex( timeseries['timestamp'], freq='H' ), inplace=True )
    timeseries.drop( labels='timestamp', axis=1, inplace=True )
    timeseries[timeseries['PV'] > 1] = 1
    return timeseries



if __name__ == '__main__':

    path = 'results'
    filepath = '/diesel_pv_batt_inv_1_'

    PH = 120
    sim_mode = 'investment'
    accuracy=0.01

    time_measure = {}

    components_list = ['demand', 'PV', 'storage', 'pp_oil_1', 'pp_oil_2', 'pp_oil_3', 'excess']
    sizing_list = ['PV', 'storage']

    start = time.time()
    initial_capacity = 0.5
    cost_dict = get_cost_dict( PH )

    timeseries = get_timeseries()
    timeseries = timeseries.iloc[:PH]

    m = diesel_only( sim_mode, timeseries, initial_capacity, cost_dict )[0]

    results = solve_and_create_results( m ,gap=accuracy)

    economic_results = lcoe.get_lcoe( m, results, components_list ).to_csv( path + filepath + 'lcoe.csv' )

    results_flows = results_postprocessing( results, components_list, time_horizon=PH )
    results_flows.to_csv( path + filepath + str( PH ) + '.csv' )

    if sim_mode == 'investment':
        sizing_df=sizing_results(results,m,sizing_list)
        sizing_df.to_csv( path + filepath + 'invest.csv' )

    end = time.time()
    time_measure[PH] = end - start
    print( time_measure[PH] )

    meta_results = processing.meta_results( m )

    with open( path + filepath + 'meta.txt', 'w' ) as file:
        file.write( str( meta_results ) )
