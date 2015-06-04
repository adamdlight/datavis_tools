import pylab
import numpy
import colorsys
import time
import matplotlib.cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

######################GRAPHICS COORDINATE TRANSFORMS######################
def data2axes(xy,ax):
    """
    Converts a pair xy=(x,y) of data coordinates to 
    axes coordinates.
    """
    disp2axes = ax.transAxes.inverted().transform
    data2disp = ax.transData.transform
    return disp2axes(data2disp(xy))


def axes2data(xy,ax):
    """
    Converts a pair xy=(x,y) of axes coordinates to 
    data coordinates.
    """
    disp2data = ax.transData.inverted().transform
    axes2disp = ax.transAxes.transform
    return disp2data(axes2disp(xy))


def fig2data(xy,fig,ax):
    """
    Converts a pair xy=(x,y) of axes coordinates to 
    data coordinates.
    """
    disp2data = ax.transData.inverted().transform
    fig2disp = fig.transFigure.transform
    return disp2data(fig2disp(xy))


def data2fig(xy,fig,ax):
    """
    Converts a pair xy=(x,y) of axes coordinates to 
    data coordinates.
    """
    disp2fig = fig.transFigure.inverted().transform
    data2disp = ax.transData.transform
    return disp2fig(data2disp(xy))



###############  Handle Utilities #######################
def get_fig(newfig=False,oplot=False,mysize=False,**kwargs):
    """
    fig = get_fig(newfig=False,oplot=False,mysize=False,**kwargs)

    Generates new figure with myfig size
    or returns figure if it exists.
    """
    if newfig:
        if mysize:
            fig = pylab.figure(figsize=myfigsize(),**kwargs) 
        else:
            fig = pylab.figure(**kwargs)
    else:
        if pylab.get_fignums() == []:     
            if mysize:
                fig = pylab.figure(figsize=myfigsize(),**kwargs) 
            else:
                fig = pylab.figure(**kwargs)
        else:
            fig = pylab.gcf()
            if not oplot:
                fig.clf()
    return fig

def new_plot(**kwargs):
    """
    ax = new_plot(**kwargs)

    Shortcut/wrapper for creating a new subplot object or 
    returning the current one.

    KWARGS are passed directly to GET_FIG.

    Uses GET_FIG to allow use of the same figure or a new one.
    
    Note that the ADD_SUBPLOT method of the figure does NOT 
    overwrite a previous 111 subplot if it already exists 
    (simply returns it).  Thus calling NEW_PLOT with keyword
    'oplot=True' returns the 111 axis in the current 
    figure if it exists.  
    """

    fig = get_fig(**kwargs)
    ax = fig.add_subplot(111)
    return ax

############### Line Plot Utilities #########################
def crosshair(ax,xc=0,yc=0,**kwargs):
    """
    Just draws a pair of lines through (xc,yc), extending to 
    the edges of the plot, with properties given by **kwargs.
    """
    if "linestyle" not in kwargs.keys():
        kwargs["linestyle"] = '--'
    if "color" not in kwargs.keys():
        kwargs["color"] = 'k'
    ax.plot([xc,xc],ax.get_ylim(),**kwargs)
    ax.plot(ax.get_xlim(),[yc,yc],**kwargs)
    return



############## Image/ndarray Visualization Utilities ##############
def two_color_linearL_quadS(H1=0.6,H2=0.13,dS=0.75,dL=0.7):
    """
    Returns a 256-color ListedColormap object.

    H1 is the hue applied to the lower half of the elements
    H2 is the hue applied to the upper half of the elements
    dS is the range of saturation values, centered on 0.5
    dL is the range of lightness values, centered on 0.5

    The hue steps from H1 to H2 in the middle of the map.
    The saturation is quadratic [x**2], with a minimum in the 
    middle of the map.
    Then lightness is linear [abs(x)], with a minimum in the middle
    of the map.


    Recall that H,L,S are all in the range [0,1].
    """
    x = numpy.arange(256)/255.
    H = numpy.zeros(256)
    H[:128] = H1
    H[128:] = H2
    S =  dS*4.0*(x-0.5)**2 + (0.5 - dS/2.)
    L = dL*2.0*numpy.abs(x-0.5) + (0.5 - dL/2.)
    rgb = numpy.zeros((256,3),float)
    for ii in numpy.arange(256):
        rgb[ii,:] = colorsys.hls_to_rgb(H[ii],L[ii],S[ii])
    
    return matplotlib.colors.ListedColormap(rgb,name='test')


def scatterimage(image,cmap=None,x=None,y=None,
                 aspect='equal'):
    """
    scatter_obj = scatterimage(image,cmap=None,x=None,y=None,
                               aspect='equal')

    Use pylab.scatter to plot an array as an image.

    Allows setting the parameters of individual pixels,
    including color, alpha, etc.

    IMAGE can be a float or int array, as well as an 
    array of RGB(A) values.
    """

    if image.ndim == 2:
        ny,nx = image.shape
        nc = 0
    elif image.ndim == 3:
        ny,nx,nc = image.shape
        if nc == 4:
            print "Image read as (nx,nx,RGBA) array."
            image = image.reshape(nx*ny,4)
        if nc == 3:
            print "Image read as (nx,nx,RGB) array."
            image = image.reshape(nx*ny,3)
    else:
        print "Image dimensions must be (nx,ny), (nx,nx,RGB), or (nx,nx,RGBA)"
        return 0

    if x is None:
        x = numpy.arange(nx)
    if y is None:
        y = numpy.arange(ny)      
    
    x,y = pylab.meshgrid(x,y)

    if cmap is None:
        cmap = matplotlib.cm.gist_heat

    ax = new_plot()
    ax.set_xlim(x.min(),x.max())
    ax.set_ylim(y.min(),y.max())
    ax.set_aspect(aspect)
    dw,dh = ax.transData.transform((x.max()-x.min(),y.max()-y.min()))
    dw /= float(nx-1)
    dh /= float(ny-1)
    m = ([(0,0),(dw,0),(dw,dh),(0,dh)],0) # marker size/shape - see scatter docstring
    ss = ax.scatter(x,y,s=dw*dh*1.2,c=image,marker=m,cmap=cmap,edgecolors='none')
    ax.set_xlim(x.min(),x.max())
    ax.set_ylim(y.min(),y.max())
    pylab.show()
    return ss

    
def imview(image,x=None,xmin=None,xmax=None,y=None,ymin=None,ymax=None,
           cmap=None,oplot=False,ax=None,has_colorbar=1,
           cbkwargs=dict(position="right",size="5%",pad=0.1),
           aspect=None,interpolation='nearest',
           bare=False,symcb=None,**kwargs):
    """
    ax,cb,im = imview(image,x=None,xmin=None,xmax=None,y=None,ymin=None,ymax=None,
                      cmap=None,oplot=False,ax=None,has_colorbar=1,
                      cbkwargs=dict(position="right",size="5%",pad=0.1),
                      aspect='equal',interpolation='nearest',**kwargs)
 
    Wrapper for imshow that implements x,y axes, proper scaling, and a colorbar.
    Uses origin='lower' to display image with (0,0) pixel in lower left corner.
    Gets tick labels from max/min values of X/Y arrays.

    Returns AX,CB,IM; an axis object, a colorbar object, and an 
    imshow object.

    IMAGE is a 2D array of intensity or amplitude values

    X is a 1D array of values for the x-axis, corresponding to the columns of IMAGE.
    Y is a 1D array of values for the y-axis, corresponding to the rows of IMAGE.
    XMIN, XMAX, YMIN, and YMAX allow the user to specify plot limits initially,
        (rather than having to adjust them later).

    CMAP is an optional colormap object to specify the image colors.  By default, 
        the gist_heat map is chosen for positive semi-definite data and a two-color
        lightness/hue map is chosen for data with both positive and negative values.

    OPLOT prevents erasure of the current objects in the given AX instance.  This allows 
        multiple images to be overlaid (usually with transparency).

    If AX is a specified axes object, put the image in that set of axes rather 
        than creating a new one.

    HAS_COLORBAR determines whether IMVIEW will automatically create space for and 
        generate a colorbar in a second axis alongside AX.

    CBKWARGS is a dictionary passed to PYLAB.COLORBAR to allow customization of the
        colorbar object.

    ASPECT allows the user to override the default choice of aspect ratio.  If left 
        unspecified, square images are plotted with aspect='equal' and rectangular
        images are plotted with aspect='auto'.

    INTERPOLATION is included here merely to override the IMSHOW default.  It has been 
        my preference to represent discrete data in a discrete (pixelated) format.

    BARE adjusts the axes relative to the figure so that the image itself fills the 
        frame.  No axes/ticks are plotted in this case.  This option is ignored if
        HAS_COLORBAR = TRUE.  

    SYMCB is a boolean flag that can force the colorbar to be symmetric about zero
        or about the mean of the image (for positive semi-definite data).
        If true, the color range is adjusted to include +/- the largest (absolute)
        value.  For positive semi-definite data, twice the largest (absolute) value
        is used for the color range and the scale is centered on the mean.
        If you wish to override this default selection, you must also specify CMAP.

    """
    
    if x is None:
        # make index array spanning number of tiles,
        # center pixels on tile points
        x = numpy.arange(image.shape[1]+1) - 0.5
    if y is None:
        y = numpy.arange(image.shape[0]+1) - 0.5
            
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    if ymin is None:
        ymin = y.min()
    if ymax is None:
        ymax = y.max()

    if aspect is None:
        if image.shape[0] == image.shape[1]:
            aspect = 'equal'
        else:
            aspect = 'auto'
    
    if cmap is None:
        if numpy.sign(image.min())*numpy.sign(image.max()) < 0:
            cmap = two_color_linearL_quadS()
            symcb = True
        else:
            cmap = matplotlib.cm.gist_heat
            symcb = False

    if symcb:                                                                   
        # if symmetric colorbar requested                                       
        # set bounds to +- same value                                           
        if "vmin" in kwargs.keys() and "vmax" in kwargs.keys(): 
            imax = kwargs["vmax"]                                            
            imin = kwargs["vmin"]
            ext = max([abs(imin),abs(imax)])
            imin = -ext
            imax =  ext
        elif "vmin" in kwargs.keys():                                          
            imin = kwargs["vmin"]                                              
            imax = -imin                      
            print "Color max set to -imin"                         
        elif "vmax" in kwargs.keys():                                          
            imax = kwargs["vmax"]                                              
            imin = -imax                      
            print "Color min set to -imax"                         
        else:                                                                   
            # otherwise symmetrize by making vmin/vmax                          
            # spaced at bigger of 2*image.min()                                 
            # or 2*image.max(), centered on the mean                            
            imax = image.max()                                                  
            imin = image.min()                                  
            if abs(imax) > abs(imin): 
                imin = -imax                                
                print "Color range set to +/- I_max"                       
            else:                                                               
                imax = -imin                                   
                print "Color range set to +/- I_min"                       
        print imin,imax
        kwargs["vmin"] = imin                                                  
        kwargs["vmax"] = imax 

    #find max/min indices
    xindx = numpy.where(numpy.logical_and(x>=xmin,x<=xmax))[0]
    ixmin = xindx[0]
    ixmax = xindx[-1]+1
    yindx = numpy.where(numpy.logical_and(y>=ymin,y<=ymax))[0]
    iymin = yindx[0]
    iymax = yindx[-1]+1
    
    #plot    
    if not ax:
        ax = new_plot(oplot=oplot)

    im = ax.imshow(image[iymin:iymax,ixmin:ixmax],origin='lower',
                   aspect=aspect,interpolation=interpolation,
                   cmap=cmap,**kwargs)
    # set up ticks
    dx = (xmax-xmin)/float(image.shape[1])
    dy = (ymax-ymin)/float(image.shape[0])
    im.set_extent((xmin-dx/2.,xmax+dx/2.,ymin-dy/2.,ymax+dy/2.))
    if has_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(**cbkwargs)
        cb = pylab.colorbar(im,cax=cax,cmap=cmap,)
    else:
        cb = None
        if bare:
            ax.set_axis_off()
            fig.subplots_adjust(top=1.0,bottom=0.0,left=0.0,right=1.0)

    pylab.show()
    return ax,cb,im


