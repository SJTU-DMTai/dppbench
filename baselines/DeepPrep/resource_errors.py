"""Keep infrastructure failures out of candidate/protocol error budgets."""


def raise_if_resource_error(error):
    if isinstance(error, MemoryError):
        raise error
    text = str(error).lower()
    if 'out of memory' in text or 'cannot allocate memory' in text:
        if isinstance(error, BaseException):
            raise error
        raise MemoryError(str(error))
