"""FTP-1 implementation exposed through the common PolicyAdapter contract."""

from robotactile_benchmark.policies.ftp1_policy import OfficialFTP1Policy

FTP1PolicyAdapter = OfficialFTP1Policy

__all__ = ["FTP1PolicyAdapter"]
