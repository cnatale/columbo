from input_transformation import to_one_hot
import torch
from lookups import set_sizes

def func(x):
    return x + 1


# def test_answer():
#     assert func(3) == 5

def test_to_one_hot():
    # should fail when no or not enough params are passed
    try:
        to_one_hot()
        assert False
    except TypeError:
        assert True

    result = torch.nn.functional.one_hot(torch.tensor([3]), num_classes=set_sizes['country'])
    print("one_hot result")
    print(result)
    print("one_hot tensor size: ")
    print(result[0].size())


