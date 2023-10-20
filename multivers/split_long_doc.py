import re

def split_long_doc(doc:str, by:str='\n\n', into:int=2) -> list:
    """
    Split a Long Document into splits (parts). It is only possible to split into 2 by now,
    and their lengths will be roughly the same (split into 2 similar length splits).
    
    Parameters:
    ----------
    doc: str, required
        Long document (string) that need to be split.
    
    by: str, default to '\n\n'
        The string to be split by.

    into: int, default to 2
        The number of sub-doc after splitting.
    
    Returns:
    ----------
    Tuple of 'into' splits.
    
    """
    split_positions = [match.start() for match in re.finditer(by, doc)]
    
    split_diffs = [abs(len(doc) - 2 * split) for split in split_positions]
    best_split_position_idx = split_diffs.index(min(split_diffs))
    best_split_position     = split_positions[best_split_position_idx]
    """
    EXPLAINATION:
    ----------
    In this case, we want our `doc` be splited into 2 smaller doc with (roughly) equal in length.
    - the `split_position` is also the length of the first doc.
    - So, the len(doc) - split_position will the the length of the second doc (but who cares?)
    
    We want length of these splits to be closest to each other
    => len(split_2)            ~ len(split_1)
    => len(doc) - len(split_1) ~ len(split_1)
    => len(doc) - 2 * len(split_1) ~ 0
    => abs(len(doc) - 2 * len(split_1)) min.
    """
    
    split_1 = doc[:best_split_position]
    split_2 = doc[best_split_position + len(by):]
    
    return [split_1, split_2]