# coding=utf-8
import re

ERROR_PATTERNS = (
    # ClickHouse prior to v19.3.3
    re.compile(r'''Code:\ (?P<code>\d+),
            \ e\.displayText\(\)\ =\ (?P<type1>[^ \n]+):\ (?P<msg>.+?),
            \ e.what\(\)\ =\ (?P<type2>[^ \n]+)
        ''', re.VERBOSE | re.DOTALL),
    # ClickHouse v19.3.3+
    re.compile(r'''Code:\ (?P<code>\d+),
            \ e\.displayText\(\)\ =\ (?P<type1>[^ \n]+):\ (?P<msg>.+)
        ''', re.VERBOSE | re.DOTALL),
)


class ParameterKeyError(Exception):
    pass


class ParameterTypeError(Exception):
    pass


class DatabaseTypeError(Exception):
    pass


class DatabaseNotExists(Exception): pass


class DatabaseError(Exception):
    '''
    Raised when a database operation fails.
    '''
    pass


class ServerError(DatabaseError):
    """
    Raised when a server returns an error.
    """

    def __init__(self, message):
        self.code = None
        processed = self.get_error_code_msg(message.decode())
        if processed:
            self.code, self.message = processed
        else:
            # just skip custom init
            # if non-standard message format
            self.message = message
            super(ServerError, self).__init__(message)

    @classmethod
    def get_error_code_msg(cls, full_error_message):
        """
        Extract the code and message of the exception that clickhouse-server generated.
        See the list of error codes here:
        https://github.com/yandex/ClickHouse/blob/master/dbms/src/Common/ErrorCodes.cpp
        """
        for pattern in ERROR_PATTERNS:
            match = pattern.match(full_error_message)
            if match:
                # assert match.group('type1') == match.group('type2')
                return int(match.group('code')), match.group('msg').strip()

        return 0, full_error_message

    def __str__(self):
        if self.code is not None:
            return "{} ({})".format(self.message, self.code)


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
