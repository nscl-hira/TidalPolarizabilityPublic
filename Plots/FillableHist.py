import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
import pandas as pd
from astropy.convolution import convolve
from astropy.convolution.kernels import Gaussian2DKernel


def WeightedMean(x, w):
  return np.average(x, weights=w)

def WeightedCov(x, y, w):
  return np.sum(w*(x - WeightedMean(x, w))*(y - WeightedMean(y, w)))/np.sum(w)

def WeightedCorr(x, y, w): 
  return WeightedCov(x, y, w) / np.sqrt( WeightedCov(x, x, w)*WeightedCov(y, y, w))


class FillableHist:

  def __init__(self, x, weights=None, bins=10, range=None, logx=False, **kwargs):
    x = np.ravel(x)
    
    self.nbins = bins
    self.histogram = np.zeros(self.nbins, dtype=np.float)

    if range is None:
      range = [np.nanmin(x), np.nanmax(x)]

    assert len(range) == 2, 'Range should be a 1D array like indicating [minx, maxx]'

    if logx:
      self.xbins = np.logspace(*np.log(range), num=self.nbins+1, base=np.e)
    else:
      self.xbins = np.linspace(*range, num=self.nbins+1)

    self.kwargs = kwargs
    self.Append(x, weights)

  def __iadd__(self, other):
    assert np.array_equal(self.xbins, other.xbins), 'Can only add histogram with equal x-bins'
    self.histogram = self.histogram + other.histogram
    return self

  def Append(self, x, weights=None):
    x = x.flatten()
    if x.shape[0] > 0:
      new_hist, self.edge = np.histogram(x, bins=self.xbins, weights=weights)
      self.histogram = self.histogram + new_hist

  def Draw(self, ax, **kwargs):
    #ax.bar(self.edge[:-1], self.histogram, width=np.diff(self.edge), align='edge', **{**kwargs, **self.kwargs})
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    mean_x = 0.5*(self.edge[:-1] + self.edge[1:])
    tot_vol = 1
    if 'normalize' in self.kwargs:
      dx = np.diff(self.edge)
      tot_vol = np.sum(dx*self.histogram)
      self.kwargs.pop('normalize')
    ax.step(mean_x, self.histogram/tot_vol, **{**kwargs, **self.kwargs})
    ax.set_xlim(min(self.edge), max(self.edge))
    ax.set_ylim(bottom=0)

  def GetMean(self):
    return WeightedMean(0.5*(self.edge[:-1] + self.edge[1:]), self.histogram)

  def GetSD(self):
    return WeightedCov(0.5*(self.edge[:-1] + self.edge[1:]), 0.5*(self.edge[:-1] + self.edge[1:]), self.histogram)




class FillableHist2D:

  def __init__(self, x, y, weights=None, bins=10, range=None, logx=False, logy=False, smooth=True, **kwargs):
    self.smooth = smooth
    x = np.ravel(x)
    y = np.ravel(y)
    assert x.shape == y.shape, 'Shape of x and y are not identical'
    
    if hasattr(bins, '__iter__'):
      if len(bins) == 2:
        self.nxbins = bins[0]
        self.nybins = bins[1]
      else:
        self.nxbins = self.nybins = bins[0]
    else:
      self.nxbins = self.nybins = bins

    self.histogram = np.zeros((self.nxbins, self.nybins), dtype=np.float)

    if range is None:
      range = [[np.nanmin(x), np.nanmax(x)], [np.nanmin(y), np.nanmax(y)]]

    assert len(range) == 2, 'Range should be a 2D array like indicating [minx, maxx], [miny, maxy]'

    if logx:
      if range[0][0] < 0: 
        range[0][0] = np.nanmin(x[x > 0])
      if range[0][1] < 0:
        range[0][1] = np.nanmax(x[x > 0])
      self.xbins = np.logspace(*np.log(range[0]), num=self.nxbins+1, base=np.e)
    else:
      self.xbins = np.linspace(*range[0], num=self.nxbins+1)

    if logy:
      if range[1][0] < 0: 
        range[1][0] = np.nanmin(y[y > 0])
      if range[1][1] < 0:
        range[1][1] = np.nanmax(y[y > 0])
      self.ybins = np.logspace(*np.log(range[1]), num=self.nybins+1, base=np.e)
    else:
      self.ybins = np.linspace(*range[1], num=self.nxbins+1)
    self.kwargs = kwargs
    self.Append(x, y, weights)

  def __iadd__(self, other):
    assert np.array_equal(self.xbins, other.xbins) and np.array_equal(self.ybins, other.ybins), 'Only histograms with equal x and y bins can be added'
    self.histogram = self.histogram + other.histogram
    return self

  def Append(self, x, y, weights=None):
    x = np.ravel(x)
    y = np.ravel(y)
    assert x.shape == y.shape, 'Shape of x and y are not identical'
    
    new_hist, self.xedge, self.yedge = np.histogram2d(x, y, bins=[self.xbins, self.ybins], weights=weights)
    self.histogram = self.histogram + new_hist

  def Draw(self, ax, **kwargs):
    X, Y = np.meshgrid(self.xedge, self.yedge)
    if self.smooth:
      Z = convolve(np.pad(self.histogram.T, pad_width=4, mode='edge'), Gaussian2DKernel(x_stddev=3))[4:-4, 4:-4]
    else:
      Z = self.histogram.T
    ax.pcolormesh(X, Y, Z, **{**kwargs, **self.kwargs})
    ax.set_xlim(min(self.xedge), max(self.xedge))
    ax.set_ylim(min(self.yedge), max(self.yedge))

  def GetMean(self, axis=0):
    if axis == 0:
      edge = self.xedge
    else:
      edge = self.yedge
    return WeightedMean(0.5*(edge[:-1] + edge[1:]), np.sum(self.histogram, axis=axis))

  def GetSD(self, axis=0):
    if axis == 0:
      edge = self.xedge
    else:
      edge = self.yedge
    return np.sqrt(WeightedCov(0.5*(edge[:-1] + edge[1:]), 0.5*(edge[:-1] + edge[1:]), np.sum(self.histogram, axis=axis)))


class PearsonCorr(FillableHist2D):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  @property
  def corr_r(self):
    self._corr_r = WeightedCorr(x=np.repeat(0.5*(self.xedge[:-1] + self.xedge[1:]), len(self.yedge)-1), y=np.tile(0.5*(self.yedge[:-1]+self.yedge[1:]), len(self.xedge)-1), w=self.histogram.flatten())
    return self._corr_r


  def Draw(self, ax, **kwargs):
    corr_text = f"{self.corr_r:2.2f}".replace("0.", ".") if abs(self.corr_r) > 0.1 else r'···'
    shay = ax.get_shared_y_axes()
    shay.remove(ax)
    #ax.clear()
    ax.set_axis_off()
    marker_size = abs(self.corr_r) * 10000 + 5000
    if abs(self.corr_r) > 0.1:
        ax.scatter([.5], [.5], marker_size, [self.corr_r], alpha=0.6, cmap="coolwarm",
                   vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(self.corr_r) * 40 + 5 + 20 if abs(self.corr_r) > 0.1 else 75
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
    ha='center', va='center', fontsize=font_size)


class FillablePairGrid:

  def __init__(self, df, weights=None, x_vars=None, y_vars=None, x_names=None, y_names=None, x_ranges=None, y_ranges=None):
    if x_vars is None:
      self.x_vars = list(df)
    else:
      self.x_vars = x_vars
    if y_vars is None:
      self.y_vars = list(df)
    else:
      self.y_vars = y_vars

    if x_names is None:
      self.x_names = self.x_vars
    else:
      self.x_names = x_names

    if y_names is None:
      self.y_names = self.y_vars
    else:
      self.y_names = y_names

    self.xvalues = df[self.x_vars].values
    self.yvalues = df[self.y_vars].values
    self.weights = weights
    if x_ranges is None:
      self.xranges = np.array([np.amin(self.xvalues, axis=0), np.amax(self.xvalues, axis=0)]).T
    else:
      x_ranges = np.atleast_2d(x_ranges)
      assert x_ranges.shape == (len(self.x_vars), 2), 'The lenght of the supplied x range disagree with number of x variables'
      self.xranges = x_ranges
    if y_ranges is None:
      self.yranges = np.array([np.amin(self.yvalues, axis=0), np.amax(self.yvalues, axis=0)]).T 
    else:
      y_ranges = np.atleast_2d(y_ranges)
      assert y_ranges.shape == (len(self.y_vars), 2), 'The lenght of the supplied y range disagree with number of y variables'
      self.yranges = y_ranges
    self.graphs = [[None for xvar in self.x_vars] for yvar in self.y_vars]

  def __iadd__(self, other):
    assert len(self.graphs) == len(other.graphs), 'Can only add pair grid if they have identical graphs'
    for idx in range(len(self.graphs)):
      for idy in range(len(self.graphs[idx])):
        self.graphs[idx][idy] += other.graphs[idx][idy]

  def map_lower(self, Fillable, **kwargs):
    for i in range(1, len(self.y_vars)):
      for j in range(0, i):
        self.graphs[i][j] = Fillable(self.xvalues[:, j], self.yvalues[:, i], weights=self.weights, range=[self.xranges[j], self.yranges[i]], **kwargs)

  def map_upper(self, Fillable, **kwargs):
    for j in range(1, len(self.x_vars)):
      for i in range(0, j):
        self.graphs[i][j] = Fillable(self.xvalues[:, j], self.yvalues[:, i], weights=self.weights, range=[self.xranges[j], self.yranges[i]], **kwargs)

  def map_diag(self, Fillable, **kwargs):
    for i in range(0, len(self.x_vars)):
      self.graphs[i][i] = Fillable(self.xvalues[:, i], weights=self.weights, range=self.xranges[i], **kwargs)

  def map(self, Fillable, **kwargs):
    for i in range(0, len(self.x_vars)):
      for j in range(0, len(self.y_vars)):
        self.graphs[j][i] = Fillable(self.xvalues[:, i], self.yvalues[:, j], weights=self.weights, range=[self.xranges[i], self.yranges[j]], **kwargs)

  def Append(self, df, weights=None):
    self.xvalues = df[self.x_vars].values
    self.yvalues = df[self.y_vars].values
    for i, row in enumerate(self.graphs):
      for j, graph in enumerate(row):
        if graph is not None:
          try:
            graph.Append(x=self.xvalues[:, j], y=self.yvalues[:,i], weights=weights)
          except Exception:
            graph.Append(x=self.xvalues[:, j], weights=weights)
        else:
          raise RuntimeError('You need to map all your graphs before appending data')
        

  def Draw(self, fontsize=40):
    self.fig, self.axes2d = plt.subplots(len(self.y_vars), len(self.x_vars))#, sharex='col', sharey='row')
    for i, row in enumerate(self.graphs):
      for j, graph in enumerate(row):
        cell = self.axes2d[i][j]

        if graph is not None:
          graph.Draw(cell)
          if i == len(self.graphs) - 1:
            cell.set_xlabel(self.x_names[j], rotation=0, fontsize=fontsize)
            cell.tick_params(axis='x', rotation=45, labelsize=fontsize)
            for x in cell.get_xticklabels():
              x.set_ha('right')
          if j == 0:
            cell.tick_params(axis='y', labelsize=fontsize)
            cell.set_ylabel(self.y_names[i], rotation=90, fontsize=fontsize)

    # join axis  
    for i, row in enumerate(self.axes2d):
      for j, ax in enumerate(row):
        ax.get_shared_x_axes().join(ax, self.axes2d[-1, j])
        # don't join y-axis of the diagonal plot if the shape is square
        if i != j and len(self.axes2d) == len(row):
          ax.get_shared_y_axes().join(ax, self.axes2d[i, 0])
        if i != len(self.axes2d) - 1:
          ax.set_xticklabels([])
        if j != 0:
          ax.set_yticklabels([])

