# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Nichole Barry, Bryna Hazelton
# Licensed under the 2-clause BSD License
"""Code for plotting EoR Limits."""

import glob
import os

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors


from eor_limits.data import DATA_PATH


def read_data_yaml(paper_name):
    """
    Read in the data from a paper yaml file.

    Parameters
    ----------
    paper_name : str
        Short name of paper (usually author_year) which corresponds to a file
        in the data directory named <paper_name>.yaml

    Returns
    -------
    dict
        Dictionary with the parsed yaml for use in the plotting code.

    """
    file_name = os.path.join(DATA_PATH, paper_name + '.yaml')

    with open(file_name, 'r') as pfile:
        paper_dict = yaml.safe_load(pfile)

    if isinstance(paper_dict['delta_squared'][0], (str,)):
        try:
            paper_dict['delta_squared'] = [float(val) for val
                                           in paper_dict['delta_squared']]
        except(ValueError):
            val_list = []
            for val in paper_dict['delta_squared']:
                if '**' in val:
                    val_split = val.split('**')
                    val_list.append(float(val_split[0])**float(val_split[1]))
                else:
                    val_list.append(float(val))
            paper_dict['delta_squared'] = val_list
    elif (isinstance(paper_dict['delta_squared'][0], (list,))
          and isinstance(paper_dict['delta_squared'][0][0], (str,))):
        for ind, elem in enumerate(paper_dict['delta_squared']):
            paper_dict['delta_squared'][ind] = [float(val) for val in elem]

    return paper_dict


def plot_eor_limits(papers=None, plot_filename='eor_limits.pdf',
                    delta_squared_range=[1e3, 1e6], colormap='Spectral_r',
                    bold_papers=None, fontsize=15):
    """
    Plot the current EoR Limits as a function of k and redshift.

    Parameters
    ----------
    papers : list of str
        List of papers to include in the plot (specified as 'author_year',
        must be present in the data folder).
        Defaults to including all papers in the data folder.
    delta_squared_range : list of float
        Range of delta squared values to include in plot (yaxis range). Must be
        length 2 with second element greater than first element.
    colormap : str
        Matplotlib colormap to use for redshift.
    plot_filename : str
        File name to save plot to.
    bold_papers : list of str
        List of papers to bold in caption.

    """
    if papers is None:
        # use all the papers. This gives weird ordering which we will fix later
        papers_sorted = False
        papers = [os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(DATA_PATH, '*.yaml'))]
    else:
        # if a list is passed in by hand, don't reorder it
        papers_sorted = True

    if bold_papers is None:
        bold_papers = []

    paper_list = []
    for paper_name in papers:
        paper_dict = read_data_yaml(paper_name)
        if paper_name in bold_papers:
            paper_dict['bold'] = True
        else:
            paper_dict['bold'] = False
        paper_list.append(paper_dict)
    if not papers_sorted:
        paper_list.sort(key=lambda paper_list: paper_list['year'])

    redshift_list = []
    for paper in paper_list:
        if paper['type'] == 'point':
            delta_array = np.array(paper['delta_squared'])
            inds_use = np.nonzero(delta_array <= delta_squared_range[1])
            redshift_list += list(np.array(paper['redshift'])[inds_use])
        else:
            for ind, elem in enumerate(paper['redshift']):
                delta_array = np.array(paper['delta_squared'][ind])
                min_delta = np.min(delta_array)
                if min_delta <= delta_squared_range[1]:
                    redshift_list += elem
    redshift_list = sorted(list(set(redshift_list)))

    norm = colors.Normalize(vmin=redshift_list[0], vmax=redshift_list[-1])
    scalarMap = cmx.ScalarMappable(norm=norm, cmap=colormap)

    fig_height = 10
    fig_width = 20
    fig = plt.figure(figsize=(fig_width, fig_height))
    legend_names = []
    lines = []
    for paper_i, paper in enumerate(paper_list):
        if paper['bold']:
            label_start = ' $\\bf{'
        else:
            label_start = ' $\\rm{'
        label_end = '}$'
        label = (label_start + '\ '.join(paper['telescope'].split(' '))
                 + '\ (' + paper['author'] + ',\ '
                 + str(paper['year']) + ')' + label_end)   # noqa
        legend_names.append(label)
        if paper['type'] == 'point':
            line = plt.scatter(paper['k'], paper['delta_squared'],
                               marker=paper['marker'],
                               c=paper['redshift'], cmap=colormap, norm=norm,
                               edgecolors='black', label=label, s=150,
                               zorder=10)
            lines.append(line)
        else:
            if not isinstance(paper['k'][0], list):
                redshifts = [paper['redshift'][0]]
                k_vals = [paper['k']]
                delta_squared = [paper['delta_squared']]
            else:
                redshifts = list(np.squeeze(paper['redshift']))
                k_vals = paper['k']
                delta_squared = paper['delta_squared']

            for ind, redshift in enumerate(redshifts):
                color_val = scalarMap.to_rgba(redshift)

                # make black outline by plotting thicker black line first
                plt.step(k_vals[ind], delta_squared[ind], c='black',
                         linewidth=paper['linewidth'] + 2, zorder=0)

                line, = plt.step(k_vals[ind], delta_squared[ind],
                                 c=color_val, linewidth=paper['linewidth'],
                                 label=label, zorder=0)
                if ind == 0:
                    lines.append(line)

    point_size = 1 / 72.  # typography standard (points/inch)
    font_inch = fontsize * point_size

    plt.rcParams.update({'font.size': fontsize})
    plt.xlabel('k ($h Mpc^{-1}$)', fontsize=fontsize)
    plt.ylabel('$\Delta^2$ ($mK^2$)', fontsize=fontsize)  # noqa
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(*delta_squared_range)
    plt.tick_params(labelsize=fontsize)
    cb = plt.colorbar(fraction=.1, pad=0.08, label='Redshift')
    cb.ax.yaxis.set_label_position('left')
    cb.ax.yaxis.set_ticks_position('left')
    cb.set_label(label='Redshift', fontsize=fontsize)
    plt.grid(axis='y')

    if fontsize > 20:
        leg_columns = 2
    else:
        leg_columns = 3

    leg_rows = int(np.ceil(len(paper_list) / leg_columns))

    legend_height = (2 * leg_rows) * font_inch

    legend_height_norm = legend_height / fig_height  # 0.25

    axis_height = 3 * fontsize * point_size
    axis_height_norm = axis_height / fig_height
    plot_bottom = legend_height_norm + axis_height_norm

    leg = plt.legend(lines, legend_names,
                     bbox_to_anchor=(0.45, legend_height_norm / 2.),
                     loc="center",
                     bbox_transform=fig.transFigure,
                     ncol=leg_columns,
                     frameon=False)

    for ind in range(len(paper_list)):
        leg.legendHandles[ind].set_color('gray')
    plt.subplots_adjust(bottom=plot_bottom)
    fig.tight_layout()
    plt.savefig(plot_filename)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--papers', type=str, nargs='+', default=None,
                        help='Papers to include on plot '
                        '(must be in data directory). Defaults to all papers '
                        'in the data directory.')
    parser.add_argument('--file', type=str,
                        help='Filename to save plot to.',
                        default='eor_limits.pdf')
    parser.add_argument('--range', type=float,
                        help='Range of Delta Squared to include on plot '
                        '(yaxis range).',
                        default=[1e3, 1e6], nargs='+')
    parser.add_argument('--colormap', type=str,
                        help='Matplotlib colormap to use.',
                        default='Spectral_r')
    parser.add_argument('--bold', type=str, nargs='+',
                        help='List of papers to bold in caption.',
                        default=None)
    parser.add_argument('--fontsize', type=int,
                        help='Font size to use.',
                        default=15)

    args = parser.parse_args()

    plot_eor_limits(papers=args.papers,
                    delta_squared_range=args.range,
                    colormap=args.colormap,
                    plot_filename=args.file,
                    bold_papers=args.bold,
                    fontsize=args.fontsize)
