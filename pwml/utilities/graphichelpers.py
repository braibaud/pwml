
import matplotlib as mat
from matplotlib import pyplot as plt
import pylab as pyl
import seaborn as sns


# Class holding static properties only
class GraphicsStatics(object):
    
    # Globals
    g_palette = []
    g_hatch = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    g_linestyle = ['-', '--', '-.', ':']

    g_landscape_fig_size = (20, 10)
    g_square_fig_size = (20, 20)
    g_portrait_fig_size = (20, 30)

    g_styles_initialized = False

    @staticmethod
    def initialize_matplotlib_styles():

        if not GraphicsStatics.g_styles_initialized:
            sns.set()
            sns.set_style('whitegrid', {'axes.facecolor': '.9'})
            plt.style.use('fivethirtyeight')
            sns.set_context('talk')

            # Assign the default palette
            GraphicsStatics.g_palette = sns.color_palette('Set2', 8)

            sns.set_palette(
                GraphicsStatics.g_palette)

            pyl.rcParams['figure.figsize'] = GraphicsStatics.g_landscape_fig_size
            plt.rcParams['figure.figsize'] = GraphicsStatics.g_landscape_fig_size
            plt.rcParams['axes.labelsize'] = 18
            plt.rcParams['axes.titlesize'] = 18

            GraphicsStatics.g_styles_initialized = True

    @staticmethod
    def get_color(index):
        return GraphicsStatics.g_palette[index % len(GraphicsStatics.g_palette)]

    @staticmethod
    def get_hatch(index):
        return GraphicsStatics.g_hatch[index % len(GraphicsStatics.g_hatch)]

    @staticmethod
    def get_linestyle(index):
        return GraphicsStatics.g_linestyle[index % len(GraphicsStatics.g_linestyle)]

    @staticmethod
    def style_plot(title, subtitle, tl_name, fig, ax, experiment_manager=None, top=0.91, tight_layout=False):

        fig.subplots_adjust(top=top)

        if title is not None:
            fig.suptitle(
                t=title, 
                fontsize=26, 
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='center',
                fontstyle='italic',
                x=(fig.subplotpars.right + fig.subplotpars.left)/2)

        if subtitle is not None:
            ax.set_title(
                label=subtitle, 
                fontdict={ 
                    'fontsize': 18,
                    'horizontalalignment': 'center'
                })

        if experiment_manager:
            experiment_id = experiment_manager.get_experiment_id()

            if experiment_manager is not None:
                fig.text(
                    x=0.98, 
                    y=1, 
                    s=experiment_id, 
                    ha='right', 
                    va='top',
                    fontsize=16,
                    fontstyle='italic',
                    bbox={ 
                        'facecolor': GraphicsStatics.get_color(6), 
                        'alpha': 0.5, 
                        'pad': 7
                    })    
        
        if tl_name is not None:
            fig.text(
                x=0.02, 
                y=1, 
                s=tl_name, 
                ha='left', 
                va='top',
                fontsize=16,
                fontstyle='italic',
                bbox={ 
                    'facecolor': GraphicsStatics.get_color(5), 
                    'alpha': 0.5, 
                    'pad': 7
                })

        if tight_layout:
            plt.tight_layout()
            
        if experiment_manager is not None:
            experiment_manager.log_chart_to_neptune(
                figure=fig, 
                name=tl_name.replace('\n', ' - '))

