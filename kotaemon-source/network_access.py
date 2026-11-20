"""Strict host/origin checks for loopback or one explicitly selected LAN IP."""

import base64
import binascii
import ipaddress
import json
import re
from urllib.parse import urlsplit


def validate_lan_ipv4(value):
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise ValueError("内网地址必须是有效的IPv4地址。") from None
    private_ranges = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
    )
    if address.version != 4 or not any(address in network for network in private_ranges):
        raise ValueError("只允许指定RFC1918私有IPv4地址用于内网访问。")
    return str(address)


def allowed_hosts(lan_ip=None):
    result = {"127.0.0.1", "localhost"}
    if lan_ip:
        result.add(validate_lan_ipv4(lan_ip))
    return result


def host_header_allowed(header, port, hosts):
    try:
        parsed = urlsplit("//" + header)
        return (
            parsed.hostname in hosts
            and parsed.port == port
            and parsed.username is None
            and parsed.password is None
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
        )
    except (ValueError, TypeError):
        return False


def origin_allowed(origin, port, hosts):
    try:
        parsed = urlsplit(origin)
        return (
            parsed.scheme == "http"
            and parsed.hostname in hosts
            and parsed.port == port
            and parsed.username is None
            and parsed.password is None
            and parsed.path == ""
            and parsed.query == ""
            and parsed.fragment == ""
        )
    except (ValueError, TypeError):
        return False


def validate_public_origin(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname
            or not re.fullmatch(r"[a-zA-Z0-9.-]+", parsed.hostname)
            or parsed.username or parsed.password or parsed.path
            or parsed.query or parsed.fragment or parsed.port not in {None, 443}):
        raise ValueError("企业入口必须是无路径和凭据的HTTPS域名，默认443端口。")
    return f"https://{parsed.hostname}" + (":443" if parsed.port else "")


def proxy_request_allowed(host, origin, client_ip, public_origin):
    if not public_origin or client_ip not in {"127.0.0.1", "::1"}:
        return False
    parsed = urlsplit(public_origin)
    return host == parsed.netloc and (not origin or origin == public_origin)


def has_app_service_principal(header):
    if not header:
        return False
    try:
        principal = json.loads(base64.b64decode(header, validate=True))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(principal, dict) or principal.get("auth_typ") != "aad":
        return False
    claims = principal.get("claims")
    if not isinstance(claims, list):
        return False
    return any(
        isinstance(claim, dict)
        and claim.get("typ") in {
            "oid", "http://schemas.microsoft.com/identity/claims/objectidentifier"
        }
        and isinstance(claim.get("val"), str)
        and bool(claim["val"].strip())
        for claim in claims
    )


def app_service_request_allowed(host, origin, public_origin, authenticated, ip_restricted=False):
    if not (authenticated or ip_restricted is True) or not public_origin:
        return False
    try:
        parsed = urlsplit(public_origin)
    except ValueError:
        return False
    return host == parsed.netloc and (not origin or origin == public_origin)


def request_within_boundary(
    host, origin, client_ip, port, hosts, public_origin=None, app_service_mode=False,
    principal_header=None, app_service_ip_restricted=False,
):
    if app_service_mode:
        return app_service_request_allowed(
            host, origin, public_origin, has_app_service_principal(principal_header),
            ip_restricted=app_service_ip_restricted,
        )
    direct_allowed = (
        host_header_allowed(host, port, hosts)
        and (not origin or origin_allowed(origin, port, hosts))
    )
    via_proxy = proxy_request_allowed(host, origin, client_ip, public_origin)
    return direct_allowed or via_proxy


class WebSocketBoundaryMiddleware:
    def __init__(self, app, **boundary_options):
        self.app = app
        self.boundary_options = boundary_options

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            headers = {}
            for name, value in scope.get("headers", []):
                name = name.lower()
                if name in {b"host", b"origin", b"x-ms-client-principal"}:
                    if name in headers:
                        await send({"type": "websocket.close", "code": 1008})
                        return
                    headers[name] = value.decode("latin-1")
            client = scope.get("client")
            if not request_within_boundary(
                host=headers.get(b"host", ""),
                origin=headers.get(b"origin"),
                client_ip=client[0] if client else "",
                principal_header=headers.get(b"x-ms-client-principal"),
                **self.boundary_options,
            ):
                await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)
