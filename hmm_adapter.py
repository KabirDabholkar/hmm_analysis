import numpy as np
import xarray as xr
from typing import Optional, Union, Any
from utils import flatten_with_lengths
from config_utils.dict_module import DictModule

accepted_types = Union[np.ndarray,xr.DataArray]

# self.adapter_specs = [
#     {
#         'method'        : 'fit',
#         'replace_with'  : partial(self.hmm.__getattribute__('fit')(
#             DictModule(
#                 module=flatten_with_lengths,
#                 in_keys=['self',['X_array', 'array']],
#                 out_keys=['X', 'lengths']
#             )
#         )
#     }
# ]




# class HMM_3dAdapter:
#     """
#     adaptor for methods of the form:
#         method(X: (trials*length,n_features), lengths= [length]*trials)
#     to the form
#         method(X: (trials,length,n_features)
#     """
#
#     def __init__(self,hmm: hmmlearn.base._AbstractHMM):
#         self.hmm = hmm
#
#
#     def __getattr__(self, item: str) -> Any:
#         return getattr(self.hmm,item)
#
#     # def __setattr__(self, key, value):
#     #     if key not in ['hmm']:
#     #         setattr(self.hmm,key,value)
#
#     def fit(
#             self,
#             X_train: accepted_types,
#             X_val: Optional[accepted_types]=None,
#     ) -> None:
#         X_train_flat, train_lengths = flatten_with_lengths(X_train)
#         kwargs = {
#             'X' : X_train_flat,
#             'lengths' : train_lengths
#         }
#         if X_val is not None:
#             assert X_train.shape[1:] == X_val.shape[1:]
#             X_val_flat, val_lengths = flatten_with_lengths(X_val)
#             kwargs = {
#                 **kwargs,
#                 'X_val': X_val_flat,
#                 'lengths_val': val_lengths
#             }
#         return self.hmm.fit(**kwargs)
#
#     def co_fit(
#             self,
#             X_in: accepted_types,
#             X_out: accepted_types,
#     ):
#         assert X_in.shape[:-1] == X_out.shape[:-1]
#         X_in_flat, lengths =  flatten_with_lengths(X_in)
#         X_out_flat, _ = flatten_with_lengths(X_out)
#         return self.hmm.co_fit(X_in_flat,X_out_flat,lengths=lengths)
#
#     def predict(self,X):
#         X_flat, lengths = flatten_with_lengths(X)
#         return self.hmm.predict(X_flat,lengths=lengths)
#
#     def score(self,X):
#         X_flat, lengths = flatten_with_lengths(X)
#         return self.hmm.score(X_flat,lengths=lengths)



def adapt_hmm_class(base_class,adapted_class_name: str):
    AdaptedHMM = type(adapted_class_name,(base_class,),{})

    def flatten_if_mode3d(X, lengths, mode3d):
        if mode3d:
            X_flat, lengths = flatten_with_lengths(X)
        else:
            X_flat = X
        return X_flat,lengths

    def _fit(
            self,
            X: accepted_types,
            lengths: list[int] = None,
            X_val: Optional[accepted_types] = None,
            lengths_val: list[int] = None,
            mode3d: bool = False,
    ) -> None:
        X_train_flat, train_lengths = flatten_if_mode3d(X, lengths, mode3d)
        kwargs = {
            'X': X_train_flat,
            'lengths': train_lengths
        }
        if X_val is not None:
            assert X.shape[1:] == X_val.shape[1:]
            X_val_flat, lengths_val = flatten_if_mode3d(X_val, lengths_val, mode3d)
            kwargs = {
                **kwargs,
                'X_val': X_val_flat,
                'lengths_val': lengths_val
            }
        super(AdaptedHMM,self).fit(**kwargs)

    def _predict(self, X, lengths=None, mode3d = False):
        X_flat,lengths = flatten_if_mode3d(X,lengths,mode3d)
        return super(AdaptedHMM,self).predict(X_flat, lengths=lengths)


    def _score(self, X, lengths=None, mode3d = False):
        X_flat,lengths = flatten_if_mode3d(X,lengths,mode3d)
        return super(AdaptedHMM,self).score(X_flat, lengths=lengths)


    def _predict_proba(self, X, lengths=None, mode3d = False):
        X_flat, lengths = flatten_if_mode3d(X, lengths, mode3d)
        out = super(AdaptedHMM, self).predict_proba(X_flat, lengths=lengths)
        if mode3d:
            out = out.reshape(*X.shape[:2],-1)
        return out

    def _co_fit(
            self,
            X_in: accepted_types,
            X_out: accepted_types,
            lengths=None,
            mode3d=False
    ):
        assert X_in.shape[:-1] == X_out.shape[:-1]
        X_in_flat, lengths = flatten_if_mode3d(X_in,lengths,mode3d)
        X_out_flat, _ = flatten_if_mode3d(X_out,[],mode3d)
        return super(AdaptedHMM,self).co_fit(X_in_flat,X_out_flat,lengths=lengths)

    AdaptedHMM.fit = _fit
    AdaptedHMM.predict = _predict
    AdaptedHMM.score = _score
    AdaptedHMM.predict_proba = _predict_proba
    AdaptedHMM.co_fit =  _co_fit
    return AdaptedHMM

from hmmlearn.hmm import BernoulliHMM, PoissonHMM, GaussianHMM
from prepare_model import CoHMM
BernoulliHMM3d = adapt_hmm_class(BernoulliHMM,'BernoulliHMM3d')
PoissonHMM3d = adapt_hmm_class(PoissonHMM,'PoissonHMM3d')
GaussianHMM3d = adapt_hmm_class(GaussianHMM,'GaussianHMM3d')
CoHMM3d =  adapt_hmm_class(CoHMM,'CoHMM3d')

if __name__ == '__main__':
    data = np.random.choice(2, size=(10, 20, 5)).astype(bool)

    # hmm = BernoulliHMM(n_components=1)
    # new_hmm = HMM_3dAdapter(hmm = hmm)
    #
    #
    # new_hmm.fit(
    #     data,
    #     X_val = data
    # )

    # d = DictModule(
    #     module = flatten_with_lengths,
    #     in_keys = [['X_array','array']],
    #     out_keys = ['X','lengths']
    # )
    #
    # print(
    #     d(X_array=data)
    # )

    # print(new_hmm)




    adapt_hmm = BernoulliHMM3d(n_components=1)

    adapt_hmm.fit(
        data,
        X_val=data,
        mode3d=True
    )

    print(
        adapt_hmm
    )

    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    # cfg = OmegaConf.create(
    #     """
    #     _target_            : hmm_adapter.adapt_hmm_class
    #     base_class          :
    #         _target_    : hmmlearn.hmm.BernoulliHMM
    #         _partial_   : true
    #     adapted_class_name  :  AdaptedBernoulliHMM
    #     """
    # )
    # print(instantiate(cfg))
