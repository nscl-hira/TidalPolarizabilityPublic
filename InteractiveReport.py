import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MakeSkyrmeFileBisection import LoadSkyrmeFile
from Utilities.EOSDrawer import EOSDrawer 
from StudyNSComposition import PressureComposition

def DrawEOS(row, ax, xlim=[1e-2, 1e4], ylim=[1e-4, 1e4], **kwargs):
    drawer.DrawEOS(df=row, ax=ax, xlim=xlim, ylim=ylim, **kwargs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$Energy\ Density\ (MeV\ fm^{-3})$')
    ax.set_ylabel(r'$Pressure\ (MeV\ fm^{-3})$')
    if 'xname' in kwargs:
        ax.set_xlabel(kwargs['xname'])
    if 'yname' in kwargs:
        ax.set_ylabel(kwargs['yname'])

def AnalysisEOS(name):
    mass, radius, pressure

def update_energy(names):
    energy_ax.clear()
    DrawEOS(df.ix[names], energy_ax)
    energy_ax.set_title(" ".join(names))

def HighlightPoints(names):
    hover_x = [df['R(1.4)'].ix[names]]
    hover_y = [df['lambda(1.4)'].ix[names]]
    highlight.set_offsets(np.c_[hover_x, hover_y])

def HighlightLine(names):
    temp = drawer.DrawEOS(df.ix[[names]], ax=pressure_ax, xname='rho/rho0', yname='GetPressure', color=['grey']*6)
    highlight_x, highlight_y = [], []
    for line in temp[names]:
        data = line.get_data()
        highlight_x = highlight_x + data[0].tolist()
        highlight_y = highlight_y + data[1].tolist()
        pressure_ax.lines.remove(line)

    highlight_line.set_data(highlight_x, highlight_y)
    
    

def update_annot(ind, names):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join(names))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)
    HighlightPoints(names) 

def update_annot_lines(name, line, ind):
    x, y = line.get_data()
    annot_lines.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    text = "{}".format(" ".join(name))
    annot_lines.set_text(text)
    annot_lines.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            names = [df.index.values[ind['ind'][0]]]
            update_energy(names)
            update_annot(ind, names)
            HighlightLine(names[0])

            annot.set_visible(True)
            highlight.set_visible(True)
            highlight_line.set_visible(True)
            pressure_fig.canvas.draw_idle()
            energy_fig.canvas.draw_idle()
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                highlight.set_visible(False)
                highlight_line.set_visible(False)
                pressure_fig.canvas.draw_idle()
                energy_fig.canvas.draw_idle()
                fig.canvas.draw_idle()


def hover_lines(event):
    vis = annot_lines.get_visible()
    if event.inaxes == pressure_ax:
        for names, lines in line_list.iteritems():
            for line in lines:
                cont, ind = line.contains(event)
                if cont:
                    update_annot_lines([names], line, ind)
                    update_energy([names])
                    HighlightPoints([names])
                    HighlightLine(names)

                    highlight.set_visible(True)
                    highlight_line.set_visible(True)
                    annot_lines.set_visible(True)
                    pressure_fig.canvas.draw_idle()
                    energy_fig.canvas.draw_idle()
                    fig.canvas.draw_idle()
                    break
                else:
                    if vis:
                        annot_lines.set_visible(False)
                        highlight.set_visible(False)
                        highlight_line.set_visible(False)
                        energy_fig.canvas.draw_idle()
                        fig.canvas.draw_idle() 
                        pressure_fig.canvas.draw_idle()
            else:
                continue
            break

filename = 'Results/Orig_mm2.17.csv'

def onclick(event):
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            names = [df.index.values[ind['ind'][0]]]
            fig, ax2 = plt.subplots()
            PressureComposition(ax2, names[0], filename) 
            plt.show()

def onclick_lines(event):
    if event.inaxes == pressure_ax:
        for names, lines in line_list.iteritems():
            for line in lines:
                cont, ind = line.contains(event)
                if cont:
                    fig, ax2 = plt.subplots()
                    PressureComposition(ax2, names, filename) 
                    plt.show()

if __name__ == '__main__':
    df = LoadSkyrmeFile(filename)
    drawer = EOSDrawer(df)

    pressure_fig, pressure_ax = plt.subplots()
    line_list = drawer.DrawEOS(ax=pressure_ax, xname='rho/rho0', yname='GetPressure', xlim=[1e-8, 6], ylim=[1e-2, 1e3], zorder=1)
    pressure_ax.set_yscale('log')
    pressure_ax.set_xlabel(r'$\rho/\rho_{0}$')

    highlight_line_list = drawer.DrawEOS(df=df.iloc[[0]], ax=pressure_ax, xname='rho/rho0', yname='GetPressure', color=['grey']*6)
    highlight_x, highlight_y = [], []
    for line in highlight_line_list[df.index.values[0]]:
        data = line.get_data()
        highlight_x = highlight_x + data[0].tolist()
        highlight_y = highlight_y + data[1].tolist()
        pressure_ax.lines.remove(line)

    highlight_line, = pressure_ax.plot(highlight_x, highlight_y, color='grey', zorder=10, linewidth=5)
    highlight_line.set_visible(False)
    

    annot_lines = pressure_ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot_lines.set_visible(False)
    pressure_fig.canvas.mpl_connect("motion_notify_event", hover_lines)
    pressure_fig.canvas.mpl_connect("button_press_event", onclick_lines)

    fig, ax = plt.subplots()
    sc = ax.scatter(df['R(1.4)'], df['lambda(1.4)'], marker='o')
    ax.set_xlim([7, 16])
    ax.set_ylim([0, 1500])
    
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    hover_x, hover_y = [], []
    highlight = ax.scatter(hover_x, hover_y, facecolor=None, edgecolor='r')
    highlight.set_visible(False)
    
    energy_fig, energy_ax = plt.subplots()
    DrawEOS(df.ix[[df.index.values[0]]], energy_ax)
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
