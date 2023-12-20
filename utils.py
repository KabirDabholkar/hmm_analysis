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

normalise = lambda array,axis=None: array/array.sum(axis=axis)


if __name__ == '__main__':
    class A:
        def __init__(self):
            return
    M = A()
    set_attributes(A,**{'a':1,'b':2})
