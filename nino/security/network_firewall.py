"""Network & SSRF Firewall: Boundary protection against rogue external and internal network requests."""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger("nino.security.network_firewall")


class FirewallBlockedException(Exception):
    """Raised when an outbound request is blocked by the network firewall."""
    pass


# Private, loopback, multicast, and cloud metadata network ranges
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),          # Current network
    ipaddress.ip_network("10.0.0.0/8"),          # RFC 1918 Private
    ipaddress.ip_network("100.64.0.0/10"),       # Carrier-grade NAT
    ipaddress.ip_network("127.0.0.0/8"),        # Loopback
    ipaddress.ip_network("169.254.0.0/16"),      # Link-local / Cloud metadata (AWS/GCP/Azure: 169.254.169.254)
    ipaddress.ip_network("172.16.0.0/12"),       # RFC 1918 Private
    ipaddress.ip_network("192.0.0.0/24"),        # IETF Protocol assignments
    ipaddress.ip_network("192.0.2.0/24"),        # TEST-NET-1
    ipaddress.ip_network("192.168.0.0/16"),      # RFC 1918 Private
    ipaddress.ip_network("198.18.0.0/15"),       # Benchmark testing
    ipaddress.ip_network("198.51.100.0/24"),     # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),      # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),         # Multicast
    ipaddress.ip_network("240.0.0.0/4"),         # Reserved / Future use
    ipaddress.ip_network("255.255.255.255/32"),  # Broadcast
    # IPv6 ranges
    ipaddress.ip_network("::/128"),              # Unspecified
    ipaddress.ip_network("::1/128"),             # Loopback
    ipaddress.ip_network("fc00::/7"),            # Unique local address (ULA)
    ipaddress.ip_network("fe80::/10"),           # Link-local
    ipaddress.ip_network("ff00::/8"),            # Multicast
]

# Explicitly blocked hostnames
BLOCKED_HOSTNAMES = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "metadata.google.internal",
    "instance-data",
}


class NetworkFirewall:
    """Outbound & Inbound Network Firewall enforcing strict SSRF and protocol policies."""

    def __init__(
        self,
        allow_internal_loopback: bool = False,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        allowed_schemes: tuple[str, ...] = ("http", "https"),
    ):
        self.allow_internal_loopback = allow_internal_loopback
        self.allowed_domains = set(d.lower() for d in allowed_domains) if allowed_domains else None
        self.blocked_domains = set(d.lower() for d in blocked_domains) if blocked_domains else set()
        self.allowed_schemes = set(s.lower() for s in allowed_schemes)

    def is_ip_blocked(self, ip_str: str) -> bool:
        """Check if an IP address belongs to any blocked/private network."""
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return True  # Invalid IP representation is blocked

        if self.allow_internal_loopback and ip.is_loopback:
            return False

        for net in BLOCKED_IP_NETWORKS:
            if ip in net:
                return True
        return False

    def validate_url(self, url: str) -> str:
        """Inspect and validate a target URL against SSRF and network security policies.

        Raises:
            FirewallBlockedException: If URL violates protocol, resolves to a private IP,
                                      or is blocked by domain policy.
        """
        if not url or not isinstance(url, str):
            raise FirewallBlockedException("Firewall blocked request: Empty or invalid URL.")

        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower()

        # 1. Enforce allowed schemes (strictly http/https, blocking file://, gopher://, dict://)
        if scheme not in self.allowed_schemes:
            raise FirewallBlockedException(
                f"Firewall blocked request: Scheme '{scheme}' is prohibited. Allowed: {list(self.allowed_schemes)}."
            )

        hostname = parsed.hostname
        if not hostname:
            raise FirewallBlockedException("Firewall blocked request: Missing hostname in URL.")

        hostname_lower = hostname.lower()

        # 2. Check explicitly blocked hostnames (e.g. localhost, cloud metadata)
        if not self.allow_internal_loopback:
            if hostname_lower in BLOCKED_HOSTNAMES:
                raise FirewallBlockedException(
                    f"Firewall blocked SSRF attempt: Access to internal/loopback hostname '{hostname}' is blocked."
                )

        if hostname_lower in self.blocked_domains:
            raise FirewallBlockedException(
                f"Firewall blocked request: Domain '{hostname}' is in blocked domain list."
            )

        # 3. Check domain allowlist if configured
        if self.allowed_domains is not None:
            domain_allowed = any(
                hostname_lower == domain or hostname_lower.endswith(f".{domain}")
                for domain in self.allowed_domains
            )
            if not domain_allowed:
                raise FirewallBlockedException(
                    f"Firewall blocked request: Domain '{hostname}' is not in approved allowlist."
                )

        # 4. Resolve DNS and check resolved IP addresses against private/SSRF ranges
        try:
            addr_info = socket.getaddrinfo(hostname, None)
        except socket.gaierror as e:
            raise FirewallBlockedException(f"Firewall DNS resolution failed for '{hostname}': {e}")

        resolved_ips = set()
        for item in addr_info:
            ip = item[4][0]
            resolved_ips.add(ip)

        for ip in resolved_ips:
            if self.is_ip_blocked(ip):
                raise FirewallBlockedException(
                    f"Firewall blocked SSRF attempt: Target host '{hostname}' resolves to restricted IP '{ip}'."
                )

        logger.debug("Firewall approved outbound URL '%s' (Resolved: %s)", url, resolved_ips)
        return url


# Global singleton instance with standard SSRF protection
network_firewall = NetworkFirewall()
