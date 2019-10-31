import matplotlib.pyplot as plt
import matplotlib as mpl

from .io import create_directories

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
mpl.rcParams['legend.framealpha'] = 0.0


def get_new_fig_and_ax(fig_id, width=9.0, height=6.0):
    fig = plt.figure(fig_id)
    ax = fig.gca()
    fig.set_size_inches(width, height)
    fig.tight_layout(pad=1.5 * mpl.rcParams['font.size']/10.0)
    return fig, ax


def save_fig(fig, path, **kwargs):
    create_directories(path)
    fig.savefig(path, dpi=300, transparent=True,
                bbox_inches='tight', **kwargs)
