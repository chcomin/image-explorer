import ipywidgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import itertools
import bisect
import pickle
import copy

class InteractiveExperimenter:
    
    def __init__(self, parameters, func, pre_compute=False, continuous_update=True, img_selector=None,
                 figsize=(9.5,8), cmap='gray'):
        """This class lets you explore the output image returned by function `func` for all possible 
        combinations of the parameters indicated in `parameters`. Since an interactive matplotlib plot is 
        shown in a Jupyter notebook, you need to include the magic command ``%matplotlib notebook`` somewhere
        in your notebook before calling this class.

        If `pre_compute` is True, `func` will be called for every possible combination of parameters
        before showing the plot. This can make the plot much more responsive in case the function
        is computation intensive.

        After instantiating the class, call the method run() for showing the plot. 

        Parameters
        ----------
        parameters : dict
            The parameters to use in the interactive plot. Must be in the format 
            {'par1_name': par1_values, 'par2_name': par2_values, 'par3_name': par3_values}, where par*_values is
            a list of values.
        func : callable
            The function that will be used for transforming an input image. The parameters must have the same
            name as in the `parameters` dict.
        pre_compute : bool
            Wether to pre-compute the output values. See above.
        continuous_update : bool
            If True, the plot is updated as the slider is moved. If False, the plot is only updated after 
            releasing the slider. Set to False if the computation of `func` takes too long and values haven't
            been pre-computed.
        img_selector : callable
            When provided, it can be used for exploring a 3D image or a collection of images. The provided
            function is called with the same arguments as `func` and must return a 2D image. For instance, 
            one can define it as ``lambda img, idx, **kwargs: img[idx]`` to be able to view slices of a 
            3D image.
        figsize : tuple
            The size of the figure
        cmap : string
            The colormap to use.
        """

        if img_selector is None:
            # Create function that accepts an image and any number of args and kwargs and returns
            # the input image
            img_selector = lambda img, *_, **__: img
        
        self.params = parameters
        self.func = func
        self.pre_compute = pre_compute
        self.continuous_update = continuous_update
        self.img_selector = img_selector
        self.figsize = figsize
        self.cmap = cmap
        
        self.fig = None
        self.plot = None
        self.img = None
        self.img_output = None
        self.show_warning = False
        self.all_img_output = None
            
        self._create_widgets()
        
    def _show(self):
        """Show the plot and all widgets"""
        
        widget_out = self.widget_out
        display(widget_out)
        
        widget_menu = ipywidgets.VBox([self.widget_alpha, self.widget_range[0], self.widget_range[1], 
                                       self.widget_fix_int, self.widget_overlay])
        widget_params = ipywidgets.VBox(list(self.widgets_params.values()))
        widgets = ipywidgets.HBox([widget_menu, widget_params])
        widgets = ipywidgets.VBox([widgets, self.widget_warning])
        with widget_out:
            fig = plt.figure(figsize=self.figsize)
            display(widgets)
            
        img_output = self.img_output
        img_min = img_output.min()
        img_max = img_output.max()
            
        ax = fig.add_subplot(111)
        plot = ax.imshow(img_output, vmin=img_min, vmax=img_max, cmap=self.cmap)   
        widget_out.clear_output(wait=True)
        
        self.fig = fig
        self.plot = plot
        
        self._change_widget_att(self.widget_range[0], f'{img_min:.2f}', self._update_display, 'value')
        self._change_widget_att(self.widget_range[1], f'{img_max:.2f}', self._update_display, 'value')
                    
    def run(self, img, **kwargs):    
        """Show the interactive plot displaying the result of processing image `img`. If the class was 
        instantiated with pre_compute=True, process the image using all possible parameters combinations 
        before showing the plot. Notice that this can take a long time.

        Additional keyword arguments are passed to the function that will be used for processing the image.
        Thus, you can pass additional, fixed, parameters to the function besides those indicated when 
        instantiating this class.
        
        Parameters
        ----------
        img : ndarray or list
            A numpy array representing the image to be processed. Can also be a 3D image or a list of images
            if an image selector was provided when instantiating the class.
        """
        
        img = copy.deepcopy(img)
        self.img = img
        self.func_kwargs = kwargs
        self.widget_out.clear_output(wait=True)
        if self.pre_compute:
            self._pre_compute_outpus(img, **kwargs)
        
        self._update_output(img)
        self._show()
                
    def _create_widgets(self):
        """Create menu and parameter widgets"""
        
        self.widget_out = ipywidgets.Output(layout={'border': '1px solid black'})
        
        self._menu_widgets()
        widgets_params = {}
        for name, param in self.params.items():
            init_val = param[(len(param)-1)//2]
            w = ipywidgets.SelectionSlider(options=param, value=init_val, description=f'{name}:',
                                        continuous_update=self.continuous_update, readout=True)
            w.observe(self._update_display, names='value')
            widgets_params[name] = w
            
        self.widgets_params = widgets_params
        
    def _menu_widgets(self):
        """Create menu widgets"""
        
        # Alpha blending
        widget_alpha = ipywidgets.FloatSlider(value=0., min=0, max=1.0, step=1, description='Alpha:',
                                   continuous_update=self.continuous_update, readout=True, readout_format='.1f')
        
        # Min and max intensities
        widget_min = ipywidgets.FloatText(value=0, step=1, description='vmin:')
        widget_max = ipywidgets.FloatText(value=255, step=1, description='vmax:')     
        
        # Set fixed intensity
        widget_fix_int = ipywidgets.Checkbox(value=False, description='Fix range', indent=True)  
        
        # Set overlay
        widget_overlay = ipywidgets.Checkbox(value=False, description='Overlay mode', indent=True)    
        
        # Warnings
        widget_warning = ipywidgets.Label(value='')
        
        widget_alpha.observe(self._update_display, names='value')
        widget_min.observe(self._update_display, names='value')
        widget_max.observe(self._update_display, names='value')
        widget_fix_int.observe(self._update_display, names='value')
        widget_overlay.observe(self._update_display, names='value')
        
        self.widget_alpha = widget_alpha
        self.widget_range = [widget_min, widget_max]
        self.widget_fix_int = widget_fix_int
        self.widget_overlay = widget_overlay
        self.widget_warning = widget_warning
        
    def _get_func_parameters(self):
        """Get parameters values that will be used for the next call of the image processing function (self.func). The values
        are obtained from the current widgets states. Also sets the additional fixed parameters of the function."""

        all_func_kwargs = dict(self.func_kwargs)
        for name in self.params:
            widget = self.widgets_params[name]
            all_func_kwargs[name] = widget.value  

        return all_func_kwargs     

    def _update_output(self, img):
        """Apply the image processing function (self.func) to the input image"""

        func_kwargs = self._get_func_parameters()
            
        if self.pre_compute:
            img_output = self.all_img_output(**func_kwargs)
        else:
            img_output = self.func(img, **func_kwargs)
        self.img_output = np.array(img_output)
        
    def _update_display(self, change):
        """This function is called when a widget changes. It is the main function used for updating the
        plot given the current widget states and function parameters.
        
        TODO: Modularize this function for improving maintenance. 
        """
        
        self.widget_warning.value = ''
        
        plot = self.plot
        if change['owner'] in self.widgets_params.values():
            param_change = True
        else:
            param_change = False
            
        if param_change:
            self._update_output(self.img)
        img_output = self.img_output

        img = self.img_selector(self.img, **self._get_func_parameters())
    
        if self.widget_overlay.value:
            self._change_widget_att(self.widget_alpha, 0.1, self._update_display, 'step')
            self._change_widget_att(self.widget_fix_int, True, self._update_display, 'disabled')
            img_output = self.blend(img, img_output, self.widget_alpha.value)                                
        else:
            self._change_widget_att(self.widget_alpha, 1, self._update_display, 'step')
            self._change_widget_att(self.widget_fix_int, False, self._update_display, 'disabled')
            if self.widget_alpha.value==1:
                img_output = img
            
        plot.set_data(img_output)
        
        # Set intensity range
        widget_range = self.widget_range
        if self.widget_fix_int.value or self.widget_overlay.value:
            plot.norm.vmin = widget_range[0].value
            plot.norm.vmax = widget_range[1].value
        else:
            # Set plot range and change widget_range values
            img_min = img_output.min()
            img_max = img_output.max()
            plot.norm.vmin = img_min
            plot.norm.vmax = img_max
            self._change_widget_att(widget_range[0], f'{img_min:.2f}', self._update_display, 'value')
            self._change_widget_att(widget_range[1], f'{img_max:.2f}', self._update_display, 'value')
        
    def _change_widget_att(self, widget, value, cb, name):
        
        widget.unobserve(cb, names=name)
        widget.__setattr__(name, value)
        widget.observe(cb, names=name)
        
    def blend(self, background, overlay, alpha):
        """Blend two images"""
        
        if overlay.max()>1:
            self.widget_warning.value = 'Warning, expected largest value of overlay image to be 1.'

        #b_max = background.max()
        #img_out = (1. - alpha)*b_max*overlay + alpha*background
        if background.ndim==2:
            background = np.tile(background[..., None], 3)  # Three channel grayscale
        if overlay.ndim==2:
            overlay_color = np.zeros((*overlay.shape, 3))
            overlay_color[:, :, 0] = 255*overlay

        mask = overlay==1
        img_out = background.copy()
        img_out[mask] = (1. - alpha)*overlay_color[mask] + alpha*background[mask]
        img_out = np.round(img_out).astype(np.uint8)
               
        return img_out
    
    def _pre_compute_outpus(self, img, **kwargs):
        """Call the image processing function (self.func) for all parameters combinations and save the
        results"""

        rc = ResultCompStor(self.params, self.func, **kwargs)
        print('Precomputing values...')
        rc.run(img)
            
        self.all_img_output = rc    
        
class ResultCompStor:
    
    def __init__(self, parameters, func, **kwargs):
        """Utility class for pre-computing the result of a function for many parameters combinations.
        After the computation, the instance of the class can be called exactly as one would call the
        original function.

        Argument `parameters` can either be a dictionary or a list. If it is a dictionary of the form
        {'par1_name': par1_values, 'par2_name': par2_values, 'par3_name': par3_values}, the cartesian
        product of the parameters will be used for combining them. That is, all possible combinations
        of the parameters are generated. If it is a list, it must have the format [par_names, combinations],
        where `par_names` is a list containing parameter names and `combinations` is a list where each
        element contains a parameter combination such as (par1_value, par2_value, par3_value).

        For instance, if `parameters` is the following dictionary:
        {'par_name1': [1, 3, 5, 7], 'par_name2': [2, 4], 'par_name3': [0.5, 1.5, 3.5]}, the function
        `func` will be calculated for the parameters [(1, 2, 0.5), (1, 2, 1.5), (1, 2, 3.5), (1, 4, 0.5), ...].
        Then, an instance of the class, say `rc`, can be called as

        rc(par_name1=3, par_name2=4, par_name3=0.5).

        Notice that the instance must be called using keyword arguments.

        Parameters
        ----------
         parameters : dict or list
            The parameters to use. See above.
        func : callable
            The function that will be called for all parameters combinations.
        **kwargs
            Additional keyword arguments are passed to `func`.
        """

        self.params = parameters
        self.func = func
        self.kwargs = kwargs
        self.indexed_param_combinations = None

        self._prepare_parameters()
        
    def __call__(self, **kwargs):
        
        ind_params = []
        for name, value in kwargs.items():
            if name in self.params:
                try:
                    ind_param = self._closest(self.params[name], value)
                except ValueError:
                    raise ValueError('The parameter value was not found.')
                ind_params.append(ind_param)

        ind_params = tuple(ind_params)
        if ind_params in self.results:
            return self.results[ind_params]
        else:
            raise ValueError('Function was not computed for the requested parameters.') 
                   
    def run(self, input):
        """Apply the function passed to the class instance to `input` using all combinations of the
        parameters. 

        After calling this method, attribute .results will contain all the calculated results.
        
        Parameters
        ----------
        input
            The input.
        """
    
        parameters = self.params

        results = {}
        for comb_idx, comb in enumerate(self.indexed_param_combinations):

            indices, params = zip(*comb)
            params = dict(zip(parameters.keys(), params))
            func_kwargs = {**params, **self.kwargs}
            if comb_idx%10==0:
                print(f'Current params: {func_kwargs}')
            output = self.func(input, **func_kwargs)     
            results[indices] = output
            
        self.results = results
        
        print('Done!')

    def _prepare_parameters(self):
        """Create list of parameters combinations"""

        parameters = self.params
        if isinstance(parameters, (list, tuple)):
            names, combinations = parameters
            param_combinations = combinations
            values = zip(*combinations)
            par_dict = {}
            for name, par_values in zip(names, values):
                par_dict[name] = sorted(list(set(par_values)))
            parameters = par_dict
            self.params = parameters

            indexed_param_combinations = []
            for comb in combinations:
                one_indexed_par_comb = []
                for name, par_val in zip(names, comb):
                    ind_par = self._closest(parameters[name], par_val)
                    one_indexed_par_comb.append((ind_par, par_val))
                indexed_param_combinations.append(tuple(one_indexed_par_comb))

        elif isinstance(parameters, dict):
            # Sorte parameters values
            par_dict = {}
            for name, values in parameters.items():
                par_dict[name] = sorted(values)
            self.params = par_dict
            indexed_parameters = {}
            for name, values in parameters.items():
                indexed_parameters[name] = zip(range(len(values)), values)    
            indexed_param_combinations = list(itertools.product(*indexed_parameters.values()))   
        else:
            raise ValueError('Argument `parameters` must be a list or dictionary.')

        self.indexed_param_combinations = indexed_param_combinations
        
    def save(self, filename):
        """Save the instance on the disk."""
        
        state = {'params': self.params, 'func': self.func,
                 'kwargs': self.kwargs, 'results': self.results}
        pickle.dump(state, open(filename, 'wb'))
     
    @classmethod
    def load(cls, filename):
        """Load an instance of the class from the disk."""
        
        state = pickle.load(open(filename, 'rb'))
        rc = cls(state['params'], state['func'], **state['kwargs'])
        rc.results = state['results']
        return rc
            
    def _closest(self, l, v):
        """Find the value in list `l` that is most similar to `v` using binary search"""

        ind = bisect.bisect_left(l, v)
        if ind>=len(l) or l[ind]!=v:
            # Not found
            raise ValueError('Bisect did not find the requested element')
        else:
            return ind