class HashableDict(dict):
    """ This class implements a hash method for dictionaries, to be used as
    keys for other dictionaries which is convenient for storing our experiment
    results.

    This dictionary cannot be changed after it has been used as a key the first
    time. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                value = HashableDict(value)
                self[key] = value

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(frozenset(self.items()))
