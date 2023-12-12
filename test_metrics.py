import numpy as np
import similarity
from omegaconf import DictConfig, OmegaConf

# generate some random data
X, Y = np.random.randn(100, 30), np.random.randn(100, 30)

# # make a particular measure
# measure = similarity.make("measure.procrustes")
# score = measure.fit_score(X, Y)
#
# print(
#     score
# )

# make all the measures
measures = similarity.make("measure")

measure_configs = similarity.make("measure", return_config=True)
# print(
#     OmegaConf.to_yaml(measure_configs)
# )

netrep_ids = [k for k,cfg in measure_configs.items() if cfg['default_backend']=='netrep']
nonnetrep_ids = [k for k,cfg in measure_configs.items() if cfg['default_backend']!='netrep']
print(
    netrep_ids,
    nonnetrep_ids
)

cca_measure = similarity.make('backend.netrep.measure.cca')
cca_measure_call = similarity.make(
    'backend.netrep.measure.cca',
    interface = {
        'fit_score':'__call__'
    }
)
score = cca_measure.fit_score(X, Y)
score_called = cca_measure_call(X, Y)



print(
    cca_measure,
    score,
    score_called
)



# for name, measure in measures.items():
#     # all the measures have the same interface
#     score = measure.fit_score(X, Y)
#     print(f"{name}: {score}")
#

# print(
#     measures.keys()
# )

# def my_metric(x, y):
#     return x.reshape(-1) @ y.reshape(-1) / (np.linalg.norm(x) * np.linalg.norm(y))
#
# # register the function with a unique id
# similarity.register(my_metric, "measure.my_metric.fit_score")
#
# metric = similarity.make("measure.my_metric")
# score = metric.fit_score(X, Y)