class BaseOp:
    # When False, this operator only acts on training rows; rows belonging to
    # the standard held-out test set (marked by __split__ == "std_test") are
    # passed through untouched. Defaults to True so any existing or new
    # operator transforms both train and std_test rows together (necessary
    # for feature-space alignment). Override to False on operators whose
    # semantics are train-only (e.g. resampling / negative sampling /
    # filter-by-label / sample-level deduplication).
    APPLIES_TO_STD_TEST = True

    def __init__(self, name):
        self.op_type = "basic op"
        self.name = name
    
    def transform(self, df):
        raise NotImplementedError("transform method not implemented")
    
    def get_op_description(self):
        raise NotImplementedError("get_op_description method not implemented")


class TabularOp(BaseOp):
    def __init__(self, name):
        super().__init__(name)
        self.op_type = "tabular op"


class TextOp(BaseOp):
    def __init__(self, name):
        super().__init__(name)
        self.op_type = "text op"


class RecOp(BaseOp):
    def __init__(self, name):
        super().__init__(name)
        self.op_type = "rec op"

class ImageOp(BaseOp):
    def __init__(self, name):
        super().__init__(name)
        self.op_type = "image op"