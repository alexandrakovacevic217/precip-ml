import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



def add_feat_plt(ax, ylabels_left=False, ylabels_top=False, ylabels_right=False, ylabels_bottom=False):
    extent = [-10, 40, 30, 50]
    ax.coastlines(resolution="auto", color="k")
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(color="k", linestyle="--", draw_labels=True)
    ax.add_feature(cfeature.LAND, facecolor=(0.8, 0.8, 0.8))

    try:
        gl.top_labels = ylabels_top
        gl.left_labels = ylabels_left
        gl.bottom_labels = ylabels_bottom
        gl.right_labels = ylabels_right
    except Exception:
        gl.xlabels_top = ylabels_top
        gl.ylabels_left = ylabels_left
        gl.xlabels_bottom = ylabels_bottom
        gl.ylabels_right = ylabels_right

    gl.xlines = False
    gl.ylines = False
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 40))
    gl.ylocator = mticker.FixedLocator(np.arange(-60, 60, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 13, "color": "gray"}
    gl.ylabel_style = {"size": 13, "color": "gray"}


def plot_fld(ds, fldplt):
    fig, axs = plt.subplots(
        nrows=1,
        ncols=1,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(12, 7),
    )
    cs = axs.contourf(ds.lon, ds.lat, fldplt, transform=ccrs.PlateCarree(), cmap="jet", extend="both")
    plt.colorbar(cs, ax=axs, orientation="horizontal")
    add_feat_plt(axs, ylabels_left=True, ylabels_top=True)
    plt.show()
