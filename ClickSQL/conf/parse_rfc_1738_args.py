# coding=utf-8
import re
from typing import Any, Union
from urllib.parse import unquote


class ArgumentError(Exception):
    """Raised when an invalid or conflicting function argument is supplied.

    This error generally corresponds to construction time state errors.

    """
    pass


def _rfc_1738_quote(text: str):
    return re.sub(r'[:@/]', lambda m: "%%%X" % ord(m.group(0)), text)


def _rfc_1738_unquote(text: str):
    return unquote(text)


def parse_rfc1738_args(name: str):
    """
    parse args and translate into dict by regex method
    ------------
    translate string format of
    'clickhouse://user:password@host:port/database'
    into
    dict version of it



    :param name: string
    :return: dict
    """
    pattern = re.compile(r'''
            (?P<name>[\w\+]+)://
            (?:
                (?P<user>[^:/]*)
                (?::(?P<password>.*))?
            @)?
            (?:
                (?:
                    \[(?P<ipv6host>[^/]+)\] |
                    (?P<ipv4host>[^/:]+)
                )?
                (?::(?P<port>[^/]*))?
            )?
            (?:/(?P<database>.*))?
            ''', re.X)

    m = pattern.match(name)
    if m is not None:
        components = m.groupdict()
        if components['database'] is not None:
            tokens = components['database'].split('?', 2)
            components['database'] = tokens[0]

        if components['user'] is not None:
            components['user'] = _rfc_1738_unquote(components['user'])

        if components['password'] is not None:
            components['password'] = _rfc_1738_unquote(components['password'])

        ipv4host: Union[str, Any] = components.pop('ipv4host')
        ipv6host: Union[str, Any] = components.pop('ipv6host')
        components['host'] = ipv4host or ipv6host

        if components['port'] is None:
            pass
        else:
            components['port'] = int(components['port'])

        return components
    else:
        raise ArgumentError(
            "Could not parse rfc1738 URL from string '%s'" % name)


if __name__ == '__main__':
    c = parse_rfc1738_args("clickhouse://test:sysy@199.199.199.199:1234/drre")
    print(c)
    pass
