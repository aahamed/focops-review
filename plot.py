'''
plot.py: Code to implement plotting for experiments
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from environment import get_threshold

plt.style.use('seaborn')

class Line( object ):
    def __init__( self, x, y, label, fmt=None ):
        self.x = x
        self.y = y
        self.label = label
        self.fmt = fmt

    def plot( self, ax ):
        if self.fmt:
            ax.plot( self.x, self.y, self.fmt, label=self.label )
        else:
            ax.plot( self.x, self.y, label=self.label )

class LineErrorBand( object ):
    def __init__( self, x, y, err, label ):
        self.x = np.array( x )
        self.y = np.array( y )
        self.err = np.array( err )
        self.label = label
        assert self.err.shape == self.y.shape
        
    def plot( self, ax ):
        ax.plot( self.x, self.y, label=self.label )
        ax.fill_between( self.x, self.y+self.err,
                self.y-self.err, alpha=0.2 )

class Plot( object ):

    def __init__( self, lines, title, x_label=None ):
        self.lines = lines
        self.title = title
        self.x_label = x_label

    def plot( self, ax ):
        ax.set_title( self.title )
        for line in self.lines:
            line.plot( ax )
        if self.x_label:
            ax.set( xlabel=self.x_label )

class Graph( object ):

    def __init__( self, plots, shape, figsize,
            filename, legend=True, sharey=False ):
        self.plots = plots
        self.shape = shape
        self.figsize = figsize
        self.filename = filename
        self.legend = legend
        self.sharey = sharey

    def plot( self ):
        rows, cols = self.shape
        fig = plt.figure( figsize=self.figsize )

        ax0 = None
        for row in range( rows ):
            for col in range( cols ):
                i = row * cols + col
                pi = self.plots[ i ]
                if i > 0 and i < 3 and self.sharey:
                    ax = fig.add_subplot( rows, cols, i+1, sharey=ax0 )
                else:
                    ax = fig.add_subplot( rows, cols, i+1 )
                if i == 0:
                    ax0 = ax
                ax.set_adjustable( 'datalim' )
                self.plots[ i ].plot( ax )
                ax.set_box_aspect( 1 )

        # add legend
        if self.legend:
            ax = fig.axes
            handles, labels = ax[1].get_legend_handles_labels()
            fig.legend( handles, labels, loc='lower left',
                    ncol=len(labels), bbox_to_anchor=(0.1, 0.02) )

        fig.subplots_adjust( wspace=0.4 )
        fig.savefig( self.filename, bbox_inches='tight' )

def get_cost_thresh( hp ):
    thresh = get_threshold( hp['env_id'], hp, hp['constraint'] )
    return thresh

def avg_r_plot( data, hp ):
    metric_id = 'AvgR'
    std_id = 'StdR'
    y = data[ metric_id ]
    x = np.arange( len(y) )
    std = data[ std_id ]
    line = LineErrorBand( x, y, std, label='focops' )
    pt = Plot( [line], 'Average Reward' )
    return pt

def avg_c_plot( data, hp ):
    lines = []
    metric_id = 'AvgC'
    std_id = 'StdC'
    y = data[ metric_id ]
    x = np.arange( len(y) )
    std = data[ std_id ]
    lines.append( LineErrorBand( x, y, std, label='focops' ) )
    thresh = get_cost_thresh( hp )
    thresh_y = [thresh] * len( y )
    lines.append( Line( x, thresh_y, label='cost_lim', fmt='k--' ) ) 
    pt = Plot( lines, 'Average Cost' )
    return pt

def smooth( arr, window=5 ):
    smooth_arr = np.zeros( arr.shape )
    for i in range( len( arr ) ):
        start = max(0, i-window)
        end = i+1
        val = np.mean( arr[start:end] )
        smooth_arr[i] = val
    return smooth_arr

def avg_speed_plot( data, hp, label='focops', show_title=False, iter_num=-1 ):
    lines = []
    metric_id = 'avg_speed_per_timestep'
    std_id = 'std_speed_per_timestep'
    y = data[ metric_id ][iter_num]
    y = smooth( y )
    x = np.arange( len(y) )
    std = data[ std_id ][iter_num]
    std = smooth( std )
    lines.append( LineErrorBand( x, y, std, label=label ) )
    title = None
    if show_title:
        title = f'Speed vs. Time iter:{iter_num+1}'
    thresh = get_cost_thresh( hp )
    eps_len = len(y)
    speed_thresh = thresh / eps_len
    thresh_y = [speed_thresh] * eps_len 
    lines.append( Line( x, thresh_y, label='cost_lim', fmt='k--' ) ) 
    pt = Plot( lines, title )
    return pt

def simple_plot( data, hp, metric_id, title, x_label=None ):
    y = data[ metric_id ]
    x = np.arange( len(y) )
    line = Line( x, y, label='focops' )
    pt = Plot( [line], title, x_label=x_label )
    return pt

def plot_data( data, hp, savename ):
    plots = []
    pt = avg_r_plot( data, hp )
    plots.append( pt )
    pt = avg_c_plot( data, hp )
    plots.append( pt )
    #plots.append( simple_plot( data, hp, 'NumV', 'Cost Violations' ) )
    #plots.append( simple_plot( data, hp, 'NumE', 'Number of Episodes' ) )
    #plots.append( simple_plot( data, hp, 'AvgEl', 'Average Episode Length',
    #    'iterations' ) )

    # create graph
    graph_dict = {
        'plots': plots,
        'shape': (1, 2),
        'filename': savename,
        'figsize': (8,4),
    }
    graph = Graph( **graph_dict )
    graph.plot()

def plot_speed_data( data, hp, savename ):
    plots = []

    label = 'focops'
    iter_num = [ 9, 49, 99 ]
    for it in iter_num:
        pt = avg_speed_plot( data, hp, show_title=True, iter_num=it )
        plots.append( pt )
    pt = avg_r_plot( data, hp )
    plots.append( pt )
    
    # create graph
    graph_dict = {
        'plots': plots,
        'shape': (1, 4),
        'filename': savename,
        'figsize': (10,10),
        'legend': True,
        'sharey': True,
    }
    graph = Graph( **graph_dict )
    graph.plot()

def load_data( log_dir ):
    files = os.listdir( log_dir )
    data_file = [ f for f in files if 'log_data' in f ][ 0 ]
    data_file = os.path.join( log_dir, data_file )
    assert data_file
    data = pickle.load( open( data_file, "rb" ) )
    return data

def load_hp( log_dir ):
    files = os.listdir( log_dir )
    hp_file = [ f for f in files if 'hyper' in f ][ 0 ]
    hp_file = os.path.join( log_dir, hp_file )
    assert hp_file
    hp = pickle.load( open( hp_file, "rb" ) )
    return hp

def test():
    # import pdb; pdb.set_trace()
    log_dir = './focops_results/Hopper-v3/exp5'
    data = load_data( log_dir )
    hp = load_hp( log_dir )
    savename = os.path.join( log_dir, 'test.png' )
    plot_data( data, hp, savename )

def plot_focops( args ):
    # import pdb; pdb.set_trace()
    log_dir = args.log_dir
    data = load_data( log_dir )
    hp = load_hp( log_dir )
    savename = os.path.join( log_dir, 'focops-plots.png' )
    plot_data( data, hp, savename )
    savename = os.path.join( log_dir, 'speed-plots.png' )
    plot_speed_data( data, hp, savename )

def main( args ):
    # test()
    plot_focops( args )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotter for FOCOPS')
    parser.add_argument('--log_dir', default='focops_results/Hopper-v3/exp0',
                        required=True, help='Directory containing experiment data')
    args = parser.parse_args()
    main(args)
