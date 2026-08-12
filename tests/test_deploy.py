"""Systemd deployment assets."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "deploy" / "install-krea2pipe-service.sh"


def rendered_unit() -> str:
    result = subprocess.run(
        [str(INSTALLER), "--print-unit"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_service_installer_has_valid_bash_syntax():
    assert INSTALLER.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(INSTALLER)], check=True)


def test_service_installer_provisions_requested_paths():
    script = INSTALLER.read_text()

    assert 'APP_HOME="/data/krea2"' in script
    assert 'MODEL_ROOT="/data/ComfyUI/models"' in script
    assert 'SERVICE_USER="azadmin"' in script
    assert "dpkg-query -W -f='${Status}' python3-dev" in script
    assert "apt-get update" in script
    assert "DEBIAN_FRONTEND=noninteractive apt-get install" in script
    assert "--no-install-recommends" in script
    assert "python3-dev" in script
    assert script.index("python3-dev") < script.index('"${APP_HOME}/bin/uv" sync')
    assert 'UV_VERSION="${UV_VERSION:-0.12.3}"' in script
    assert 'elif [[ -x "${APP_HOME}/bin/uv" ]]' in script
    assert '"${sudo_home}/.local/bin/uv"' in script
    assert "https://astral.sh/uv/${UV_VERSION}/install.sh" in script
    assert "curl --proto '=https' --proto-redir '=https' --tlsv1.2 -LsSf" in script
    assert 'wget -qO "${UV_INSTALLER}"' in script
    assert 'UV_UNMANAGED_INSTALL="${APP_HOME}/bin"' in script
    assert "UV_NO_MODIFY_PATH=1" in script
    assert "nvidia-modprobe" in script
    assert "nvidia-smi" in script
    assert 'chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${APP_HOME}"' in script
    assert "install -d -m 2770" in script
    assert '"${APP_HOME}/state"' in script
    assert 'set_config_value "state-dir" \'"/data/krea2/state"\'' in script
    assert 'set_config_value "subdir" \'"%hostname"\'' in script
    assert 'set_config_value "service-mode" "true"' in script
    assert 'set_config_value "api-host" \'"127.0.0.1"\'' in script
    assert 'set_config_value "api-port" "8787"' in script
    assert "reconcile-interval = 300" in script
    assert '"${APP_HOME}/bin/uv" sync' in script
    assert 'SERVICE_DRIVER="/usr/local/libexec/krea2pipe-service"' in script
    assert 'install -m 0755 "${BASH_SOURCE[0]}" "${SERVICE_DRIVER}"' in script
    assert 'install -m 0644 <(render_unit) "${UNIT_PATH}"' in script
    assert 'sources = ["/data/krea2/prompts/**/*.txt"]' in script
    assert "systemctl enable --now" in script


def test_generated_systemd_unit_has_one_restart_path():
    unit = rendered_unit()

    assert "User=azadmin" in unit
    assert "Group=azadmin" in unit
    assert "ExecStartPre=+/usr/local/libexec/krea2pipe-service --service-preflight" in unit
    assert "ExecStart=/usr/local/libexec/krea2pipe-service --service-run" in unit
    assert "ReadOnlyPaths=-/data/ComfyUI/models" in unit
    assert "ReadWritePaths=-/data/krea2" in unit
    assert "StartLimitIntervalSec=0" in unit
    assert "Restart=on-failure" in unit
    assert "RestartSec=30" in unit
    assert "TimeoutStartSec=infinity" in unit
    assert "OnFailure=" not in unit
    assert "RequiresMountsFor=" not in unit
    assert "NoNewPrivileges=true" in unit


def test_preflight_consolidates_mount_and_cuda_readiness():
    script = INSTALLER.read_text()
    preflight = script[script.index("service_preflight()") : script.index("service_run()")]

    assert 'mountpoint --quiet "${DATA_MOUNT}"' in preflight
    assert "systemctl start data.mount" in preflight
    assert 'timeout "${PROBE_TIMEOUT}" nvidia-modprobe' in preflight
    assert 'timeout "${PROBE_TIMEOUT}" nvidia-smi' in preflight
    assert 'timeout "${PROBE_TIMEOUT}" nvidia-modprobe -u' in preflight
    assert "-c 'import torch; torch.cuda.init()'" in preflight
    assert preflight.index("nvidia-smi") < preflight.index("nvidia-modprobe -u")
    assert 'sleep "${RETRY_INTERVAL}"' in preflight


def test_legacy_deployment_assets_are_consolidated():
    assert not (ROOT / "deploy" / "krea2pipe.service").exists()
    assert not (ROOT / "deploy" / "krea2pipe-retry.service").exists()
    assert not (ROOT / "deploy" / "krea2pipe-wait-for-cuda").exists()
