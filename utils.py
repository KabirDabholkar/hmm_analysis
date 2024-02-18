import numpy as np
from copy import deepcopy
from typing import Union
import xarray as xr
from collections.abc import Iterable
import numpy as np
import os

def indicator_func_to_matrix(size,indicator_func,dtype=bool):
    A = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            A[i,j] = indicator_func(i,j)
    return A.astype(dtype)

def make_path_if_not_exist(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def duplicate_list(item,repeats):
    return [item] * repeats

def set_attributes(object, **kwargs):
    if not len(kwargs):
        return object
    for attr,val in kwargs.items(): #zip(attributes,values):
        setattr(object,attr,val)
    return object

def copy_attributes(from_object,to_object,attributes):
    for attr in attributes:
        setattr(
            to_object,
            attr,
            getattr(from_object,attr)
        )
    return to_object

def empty_call(func):
    return func()

def select(obj,**kwargs):
    return obj.select(**kwargs)

def flatten_with_lengths(array: Union[np.ndarray,xr.DataArray]):
    array_values = array
    if isinstance(array,xr.DataArray):
        array_values = array.values

    trials,length = array.shape[:2]
    array_r = np.reshape(array_values,(-1,*array.shape[2:]))
    return array_r, [length]*trials


class HMM_Dataset():
    array = None
    def __init__(self,
                 ndarray,
                 argnames=['X', 'lengths'],
                 prep_funcs = [flatten_with_lengths]):
        assert ndarray.ndim==3
        self.array = ndarray
        self.argnames = argnames
        self.prep_funcs = prep_funcs

    def split(self,indices,axis=None):
        return tuple([HMM_Dataset(x) for x in np.split(self.array,indices,axis=axis)])
    def __call__(self):
        x = deepcopy(self.array)
        for p in self.prep_funcs:
            x=p(x)
        return dict(
            zip(
                self.argnames,
                x
            )
        )

#
# class DataArrayCoordsWithList(xr.DataArray):
#     def __init__(self):

class MultiSelectDataArray(xr.DataArray):
    caller_argnames = ('X','lengths')
    def select(self,**kwargs):
        """
        only works if resulting masks are 1-D.
        """
        if len(kwargs)==0:
            return self

        key, val = kwargs.popitem()
        vals = val if isinstance(val, Iterable) else [val]
        mask = self[key].isin(vals)
        # print(self.coords)
        # print({key:mask})
        return self.sel(**{self[key].dims[0]:mask}).select(**kwargs)
    def __call__(self,keys=None):
        keys = self.caller_argnames if keys is None else keys
        return dict(
            zip(
                keys,
                flatten_with_lengths(np.array(self))
            )
        )


setattrs = lambda target, attributes, values:  [setattr(target,attr,val) for attr,val in zip(attributes,values)]
setattrs_kwargs = lambda target,**kwargs: setattrs(target,*list(zip(*kwargs.items())))

def make_hmm_xarray(array):
    xr.DataArray(
        array
    )
#def flatten_with_lengths_xarray(xarray):



# def __init__(self,ndarray):
#     super().__init__()
#     self.array = ndarray
#
# def split(self,indices,axis=None):
#     return np.split(self,indices,axis=axis)


normalise = lambda array,axis=None: array/array.sum(axis=axis,keepdims=True)


if __name__ == '__main__':
    # class A:
    #     def __init__(self):
    #         return
    # M = A()
    # set_attributes(A,**{'a':1,'b':2})

    # a = HMM_Dataset(np.ones((3,2,2)))
    # print(a.split([1],axis=2)[0]())
    # print()
    # A = HMMData(

    A = xr.DataArray(
        np.random.uniform(size=(5,10,20)),
        dims=('trials','time','neurons'),
        coords={
            'trials':['train']*2+['test']*3,
            'neurons': ['heldin']*10+['heldout']*10

            # 'neuron_part':['heldin']*10+['heldout']*10,
        }
    )


    # A.coords['heldin'] = ('neurons',np.arange(20) < 10)
    # A.coords['heldout'] = ('neurons', np.arange(20) > 10)
    # print(A.sel(neurons='heldin'))
    # print(A.loc['train',:,'heldin'].shape)
    # print(A.sel(trials='train',neurons_part=['heldin','heldout']).shape)
    # print(A.groupby('neurons')['heldout'].shape)
    # print(A.sel(heldin=True))
    # print(A.to_dataframe(name='a'))
    # print(A.loc[:,:,'heldin':'heldout'].shape)
    # print(A())

    A = xr.DataArray(
        np.random.uniform(size=(5, 10, 20)),
        dims=('trials', 'time', 'neurons'),
        # coords={
        #     'trials': np.arange(5),
        #     'time'  : np.arange(10),
        #     'neurons': np.arange(20)
        #     # 'neuron_part':['heldin']*10+['heldout']*10,
        # }
    )

    B = xr.Dataset(
        {
            'data': A,
            'trial_type': xr.DataArray(['train'] * 2 + ['test']*3,dims='trials'),
            'neuron_type': xr.DataArray(['heldin'] * 10 + ['heldout'] * 10, dims='neurons')
        }
    )
    # print(B.where(B.trial_type == 'train').data.shape)
    # print(B.where(B.neuron_type in ['heldin','heldout']).data.shape)

    data = xr.DataArray(
        np.arange(15),
        coords={
            'trials': ['train'] * 5 + ['val'] * 5 + ['test'] * 5
        }
    )
    # print(data.sel(trials='train'))              # works
    # print(data.sel(trials=['train','test']))     # doesnt work

    data = xr.DataArray(
        np.arange(15),
        dims = ('trials'),
        coords={
            # 'trials': np.arange(15),
            'trial_split':('trials',['train'] * 5 + ['val'] * 5 + ['test'] * 5),
        }
    )
    mask1 = data['trial_split'].isin(['train','test'])
    mask2 = data.trial_split.isin(['train','test'])
    # mask =
    # print(data[mask])
    # print(mask1 & True )

    data = MultiSelectDataArray(
        np.ones((10,20,30)),
        dims = ('trials','time','neurons'),
        coords={
            # 'trials': np.arange(15),
            'trial_split':('trials',['train'] * 5 + ['val'] * 2 + ['test'] * 3),
            'neurons_split':('neurons',['heldin']*10+['heldout']*20)
        }
    )

    print(data.select(trial_split=['train','test'],neurons_split='heldin').shape) #

    data = xr.DataArray(
        np.ones((15,20,30)),
        dims = ('trials','time','neurons'),
        coords={
            'train':('trials',np.arange(15) <  5),
            'val':  ('trials',(np.arange(15) >= 5)&(np.arange(15) < 10)),
            'test': ('trials', np.arange(15) >= 10),
            'heldin':('neurons',np.arange(30)<15)
        }
    )
    print(data.sel(trials=data['train'],neurons=data['heldin']).shape)
    # print(data.where(data['train']).shape)
    # print(data.trial_split.dims[0])
    # b = {'a':1}
    # print(b.popitem())
    # print(dir(b))
    print(np.reshape(data.values,-1))
