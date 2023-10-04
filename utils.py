def duplicate_list(item,repeats):
    return [item] * repeats


def copy_attributes(from_object,to_object,attributes):
    for attr in attributes:
        setattr(
            to_object,
            attr,
            getattr(from_object,attr)
        )
    return to_object