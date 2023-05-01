def _link_on_path(path, link):
    # function adapted from:
    # https://www.geeksforgeeks.org/python-check-for-sublist-in-list/
    # Check for Sublist in List
    # Using any() + list slicing + generator expression
    res = any(path[idx : idx + len(link)] == link
            for idx in range(len(path) - len(link) + 1))
    return res

def link_on_path(path, link):
    l2 = [link[1], link[0]]
    return _link_on_path(path, link) or _link_on_path(path, l2)
