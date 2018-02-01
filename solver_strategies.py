import pandas as pd
from oemof.outputlib import views,processing
import cost_summary as lcoe
import main


def rolling_horizon(PH, CH, SH=8760):
    iter = 0
    start = 0
    stop = PH
    mode = 'simulation'
    initial_capacity=0.5

    path = 'results'
    filepath = '/diesel_pv_batt_PH120_P1_B1'

    components_list = ['demand', 'PV', 'storage', 'pp_oil_1', 'pp_oil_2', 'pp_oil_3', 'excess']

    results_list = []
    economic_list = []

    cost = main.get_cost_dict( PH )
    file = 'data/timeseries.csv'

    timeseries = pd.read_csv( file, sep=';' )
    timeseries.set_index( pd.DatetimeIndex( timeseries['timestamp'], freq='H' ), inplace=True )
    timeseries.drop( labels='timestamp', axis=1, inplace=True )
    timeseries[timeseries['PV'] > 1] = 1

    itermax = int( (SH / CH) - 1 )

    time_measure = {}

    while iter <= itermax:

        if iter == 0:
            status = True
        else:
            status = False

        feedin_RH = timeseries.iloc[start:stop]

        print( str( iter + 1 ) + '/' + str( itermax + 1 ) )

        m = main.diesel_only( mode, feedin_RH, initial_capacity, cost, iterstatus=status)[0]

        results_el = main.solve_and_create_results( m )

        results_list.append( main.results_postprocessing( results_el, components_list, time_horizon=CH ) )

        economic_list.append( lcoe.get_lcoe( m, results_el, components_list ) )

        initial_capacity = views.node( results_el, 'storage' )['sequences'][(('storage', 'None'), 'capacity')][CH - 1]

        start += CH
        stop += CH

        iter += 1

    res = pd.concat( results_list, axis=0 )
    economics = pd.concat( economic_list, axis=0,keys=range(itermax+1) )
    economics.to_csv(path+filepath+str(PH)+ '_lcoe.csv')
    res.to_csv( path + filepath + str( PH ) + '_.csv' )

    meta_results = processing.meta_results( m )

    with open( path + filepath + 'meta.txt', 'w' ) as file:
        file.write( str( meta_results ) )

if __name__ == '__main__':
    ph=120
    ch=120

    rolling_horizon(ph,ch)
