import config_utils
from config_utils.dict_module import DictSequential, DictModule
# result = config_utils.make('b','cue_configs', package="test/test1:test")
# print(result)



add = lambda a,b:a+b
subtract2 = lambda a: a-2

module1 = DictModule(
    module=add,
    in_keys=[['x','a'],['y','b']],
    out_keys=['z']
)
module2 = DictModule(
    module=subtract2,
    in_keys=[['z','a']],
    out_keys=['v']
)



x = 1
y = 2

composition = DictModule(
    module=DictSequential(module1,module2),
    in_keys=['x','y'],
    out_keys=[['v', None]]
)
print(
    composition(**{'x':x, 'y':y})
)

