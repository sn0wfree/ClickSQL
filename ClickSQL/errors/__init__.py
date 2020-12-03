# coding=utf-8


class ParameterKeyError(Exception):
    pass


class ParameterTypeError(Exception):
    pass


class DatabaseTypeError(Exception):
    pass


class DatabaseNotExists(Exception): pass


class DatabaseError(Exception):
    pass


class HeartbeatCheckFailure(Exception):
    pass


class ClickHouseTableExistsError(Exception):
    pass


class ClickHouseTableNotExistsError(Exception):
    pass


class ArgumentError(Exception):
    """Raised when an invalid or conflicting function argument is supplied.

    This error generally corresponds to construction time state errors.

    """
    pass
