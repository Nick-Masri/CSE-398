from rtree import index

def test_intersection():
    # create a new R-tree index
    idx = index.Index()

    # add some rectangles to the index
    idx.insert(1, (0, 2, 0, 2))  # rectangle 1
    idx.insert(2, (1, 3, 1, 3))  # rectangle 2

    result = list(idx.intersection((0, 1, 0, 1))) # rectangle that intersects with both rectangles in the index
    print(result)
    # assert len(result) == 2
    # assert set(result) == {1, 2}

test_intersection()
