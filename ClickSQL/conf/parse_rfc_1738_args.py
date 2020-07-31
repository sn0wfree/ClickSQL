# coding=utf-8
import re
from urllib.parse import unquote


class ArgumentError(Exception):
    """Raised when an invalid or conflicting function argument is supplied.

    This error generally corresponds to construction time state errors.

    """


def _parse_rfc1738_args(name):
    pattern = re.compile(r'''
            (?P<name>[\w\+]+)://
            (?:
                (?P<username>[^:/]*)
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

        if components['username'] is not None:
            components['username'] = _rfc_1738_unquote(components['username'])

        if components['password'] is not None:
            components['password'] = _rfc_1738_unquote(components['password'])

        ipv4host = components.pop('ipv4host')
        ipv6host = components.pop('ipv6host')
        components['host'] = ipv4host or ipv6host
        components['port'] = int(components['port'])
        return components
    else:
        raise ArgumentError(
            "Could not parse rfc1738 URL from string '%s'" % name)


def _rfc_1738_quote(text):
    return re.sub(r'[:@/]', lambda m: "%%%X" % ord(m.group(0)), text)


def _rfc_1738_unquote(text):
    return unquote(text)


class ParseRFC1738Args(object):
    @staticmethod
    def parse(rfc1738_args):
        return _parse_rfc1738_args(rfc1738_args)


if __name__ == '__main__':
    c = _parse_rfc1738_args("clickhouse://test:sysy@199.199.199.199:1234/drre")
    print(c)
    pass
