"""Tool execution permission tiers and security policy definitions for Nino."""

from enum import Enum


class PermissionTier(str, Enum):
    """Execution permission levels categorizing tools by risk, side effects, and authorization requirements.

    Policies:
    - READ_ONLY:
        Operations that strictly inspect system state without modifying anything.
        No side effects, zero risk of data corruption or leakage.
        Examples: Reading system clock, inspecting public disk space, searching local documentation.
        Policy: Can be executed autonomously without human confirmation.

    - REVERSIBLE:
        Operations that modify state or produce output, but can be cleanly rolled back or undone
        without permanent data loss or lasting side effects.
        Examples: Creating temporary files in scratch/, staging changes in a git branch, setting a session timer.
        Policy: Can be executed autonomously, with full activity audit logging and undo hooks registered.

    - SENSITIVE:
        Operations that modify persistent user data, access private personal records, or perform actions
        that cannot be automatically rolled back without human intervention.
        Examples: Modifying calendar events, sending local notifications, writing to user documents.
        Policy: Requires interactive human review/approval before execution unless explicitly whitelisted
        by user session policy.

    - RESTRICTED:
        High-risk operations involving system-level modifications, credential access, external network egress,
        destructive operations, or arbitrary shell execution.
        Examples: Spawning system shell commands (`rm`, `format`, `curl`), deleting user directories,
        accessing secret keyrings or password stores, outbound external API transmissions.
        Policy: Strictly blocked by default. Requires explicit multi-factor or per-action explicit affirmative
        consent with full parameter inspection and immutable audit trail.
    """

    READ_ONLY = "READ_ONLY"
    REVERSIBLE = "REVERSIBLE"
    SENSITIVE = "SENSITIVE"
    RESTRICTED = "RESTRICTED"
