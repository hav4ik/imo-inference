# OPD evaluation supervisor service

Install the checked-in wrapper and supervisor program definition:

```bash
install -m 0755 evaluation/supervisor/opd32b-eval.sh \
  /opt/supervisor-scripts/opd32b-eval.sh
install -m 0644 evaluation/supervisor/opd32b-eval.conf \
  /etc/supervisor/conf.d/opd32b-eval.conf
supervisorctl reread
supervisorctl update
```

The service always launches the repository's single YAML on GPUs 0 and 1. It
does not contain model-mode, DFlash, quantization, or budget overrides. Logs are
written to `/var/log/portal/opd32b-eval.log` and rotated on service start.
