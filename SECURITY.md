# Security Policy

Report vulnerabilities through the repository host's private security-advisory
feature. Do not include credentials, private datasets, or exploit details in a
public issue.

RoboTactile treats external checkouts, artifact manifests, model bundles, and
result bundles as untrusted inputs. Loaders reject source drift, symlink escape,
unknown fields, noncanonical JSON, and hash mismatch. Installation scripts do
not request credentials or download model weights.

The project does not promise network isolation, TLS termination, or sandboxing
for third-party model runtimes. Deploy those runtimes on a trusted private
network and follow their own security guidance.
