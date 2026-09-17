class RowNotFoundError(Exception):
    """Raised when a delete targets a (sub_id, session) that doesn't exist
    in the target store -- e.g. it was already removed by another action."""
