"""Systemd deployment assets."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "deploy" / "install-krea2pipe-service.sh"
UNIT = ROOT / "deploy" / "krea2pipe.service"
RETRY_UNIT = ROOT / "deploy" / "krea2pipe-retry.service"


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
    assert 'chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${APP_HOME}"' in script
    assert "install -d -m 2770" in script
    assert '"${APP_HOME}/state"' in script
    assert 'state-dir = "/data/krea2/state"' in script
    assert "service-mode = true" in script
    assert 'api-host = "127.0.0.1"' in script
    assert "api-port = 8787" in script
    assert "reconcile-interval = 300" in script
    assert '"${APP_HOME}/bin/uv" sync' in script
    assert 'sources = ["/data/krea2/prompts/**/*.txt"]' in script
    assert '"${SOURCE_ROOT}/deploy/krea2pipe-retry.service"' in script
    assert "systemctl enable --now" in script


def test_systemd_unit_uses_dedicated_home_and_model_library():
    unit = UNIT.read_text()

    assert "User=azadmin" in unit
    assert "Group=azadmin" in unit
    assert "WorkingDirectory=/data/krea2" in unit
    assert "ExecStart=/data/krea2/.venv/bin/krea2pipe" in unit
    assert "ReadOnlyPaths=/data/ComfyUI/models" in unit
    assert "ReadWritePaths=/data/krea2" in unit
    assert "RequiresMountsFor=/data/krea2 /data/ComfyUI/models" in unit
    assert "OnFailure=krea2pipe-retry.service" in unit


def test_systemd_retry_covers_failures_before_service_process_start():
    unit = RETRY_UNIT.read_text()

    assert "PartOf=krea2pipe.service" in unit
    assert "StartLimitIntervalSec=0" in unit
    assert "ExecStart=/usr/bin/systemctl start krea2pipe.service" in unit
    assert "Restart=on-failure" in unit
    assert "RestartSec=30" in unit
