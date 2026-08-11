#!/usr/bin/env bash
set -Eeuo pipefail

APP_HOME="/data/krea2"
MODEL_ROOT="/data/ComfyUI/models"
SERVICE_NAME="krea2pipe"
SERVICE_USER="azadmin"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
RETRY_UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}-retry.service"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
START_SERVICE=1
UV_VERSION="${UV_VERSION:-0.12.3}"

usage() {
    cat <<'EOF'
Usage: sudo deploy/install-krea2pipe-service.sh [--no-start]

Installs krea2pipe under /data/krea2 for the azadmin user and installs the
krea2pipe systemd service. Existing runtime configuration and generated output
are preserved. If uv is unavailable, the official Astral standalone installer
installs it under /data/krea2/bin without modifying shell profiles.

Environment:
  UV_BIN      Use a specific existing uv executable.
  UV_VERSION  Bootstrap this uv version when installation is needed (default: 0.12.3).
EOF
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

for argument in "$@"; do
    case "${argument}" in
        --no-start)
            START_SERVICE=0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "Unknown argument: ${argument}"
            ;;
    esac
done

[[ "${EUID}" -eq 0 ]] || fail "Run this installer as root, for example with sudo."

for command in chmod getent grep install mktemp rsync runuser sed sh systemctl usermod; do
    command -v "${command}" >/dev/null || fail "Required command not found: ${command}"
done

if ! command -v dpkg-query >/dev/null \
    || [[ "$(dpkg-query -W -f='${Status}' python3-dev 2>/dev/null || true)" \
        != "install ok installed" ]]; then
    command -v apt-get >/dev/null || fail \
        "python3-dev is required, but apt-get is unavailable."
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-dev
fi

UV_SOURCE=""
if [[ -n "${UV_BIN:-}" ]]; then
    [[ -x "${UV_BIN}" ]] || fail "UV_BIN is not executable: ${UV_BIN}"
    UV_SOURCE="${UV_BIN}"
elif [[ -x "${APP_HOME}/bin/uv" ]]; then
    UV_SOURCE="${APP_HOME}/bin/uv"
elif command -v uv >/dev/null; then
    UV_SOURCE="$(command -v uv)"
elif [[ -n "${SUDO_USER:-}" ]]; then
    sudo_home="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
    if [[ -x "${sudo_home}/.local/bin/uv" ]]; then
        UV_SOURCE="${sudo_home}/.local/bin/uv"
    fi
fi
[[ -d "${MODEL_ROOT}" ]] || fail "Model root does not exist: ${MODEL_ROOT}"
[[ -f "${SOURCE_ROOT}/pyproject.toml" ]] || fail "Run this script from the krea2pipe source tree."
[[ -f "${SOURCE_ROOT}/uv.lock" ]] || fail "Missing locked dependencies: ${SOURCE_ROOT}/uv.lock"

getent passwd "${SERVICE_USER}" >/dev/null || fail \
    "Required service user does not exist: ${SERVICE_USER}"
SERVICE_GROUP="$(id -gn "${SERVICE_USER}")"

for group in video render; do
    if getent group "${group}" >/dev/null; then
        usermod --append --groups "${group}" "${SERVICE_USER}"
    fi
done
runuser -u "${SERVICE_USER}" -- test -x "${MODEL_ROOT}" || fail \
    "${SERVICE_USER} cannot access model root ${MODEL_ROOT}."

install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
    "${APP_HOME}" \
    "${APP_HOME}/bin" \
    "${APP_HOME}/.cache/uv"
install -d -m 2750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
    "${APP_HOME}/state"
install -d -m 2770 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
    "${APP_HOME}/prompts"
install -d -m 2750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
    "${APP_HOME}/output"

if [[ -n "${UV_SOURCE}" ]]; then
    if [[ "${UV_SOURCE}" != "${APP_HOME}/bin/uv" ]]; then
        install -m 0755 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
            "${UV_SOURCE}" "${APP_HOME}/bin/uv"
    fi
else
    [[ "${UV_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail \
        "UV_VERSION must be a stable semantic version such as 0.12.3"
    UV_INSTALLER="$(mktemp)"
    cleanup_uv_installer() {
        rm -f -- "${UV_INSTALLER}"
    }
    trap cleanup_uv_installer EXIT
    UV_INSTALLER_URL="https://astral.sh/uv/${UV_VERSION}/install.sh"
    if command -v curl >/dev/null; then
        curl --proto '=https' --proto-redir '=https' --tlsv1.2 -LsSf \
            "${UV_INSTALLER_URL}" -o "${UV_INSTALLER}"
    elif command -v wget >/dev/null; then
        wget -qO "${UV_INSTALLER}" "${UV_INSTALLER_URL}"
    else
        fail "uv is unavailable and bootstrapping requires curl or wget."
    fi
    chmod 0644 "${UV_INSTALLER}"
    runuser -u "${SERVICE_USER}" -- env \
        HOME="${APP_HOME}" \
        UV_UNMANAGED_INSTALL="${APP_HOME}/bin" \
        UV_NO_MODIFY_PATH=1 \
        sh "${UV_INSTALLER}"
    [[ -x "${APP_HOME}/bin/uv" ]] || fail \
        "Astral's uv installer did not create ${APP_HOME}/bin/uv"
    cleanup_uv_installer
    trap - EXIT
fi
runuser -u "${SERVICE_USER}" -- "${APP_HOME}/bin/uv" --version

if systemctl is-active --quiet "${SERVICE_NAME}.service"; then
    systemctl stop "${SERVICE_NAME}.service"
fi
if systemctl is-active --quiet "${SERVICE_NAME}-retry.service"; then
    systemctl stop "${SERVICE_NAME}-retry.service"
fi

install -m 0644 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
    "${SOURCE_ROOT}/README.md" \
    "${SOURCE_ROOT}/pyproject.toml" \
    "${SOURCE_ROOT}/uv.lock" \
    "${APP_HOME}/"
install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" "${APP_HOME}/src"
rsync -a --delete "${SOURCE_ROOT}/src/" "${APP_HOME}/src/"
chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "${APP_HOME}"

CONFIG_PATH="${APP_HOME}/krea2pipe.toml"
if [[ ! -e "${CONFIG_PATH}" ]]; then
    install -m 0640 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}" \
        "${SOURCE_ROOT}/krea2pipe.toml" "${CONFIG_PATH}"
    printf '\nsources = ["/data/krea2/prompts/**/*.txt"]\n' >>"${CONFIG_PATH}"
fi

if grep -Eq '^[[:space:]]*model-root[[:space:]]*=' "${CONFIG_PATH}"; then
    sed -Ei \
        's|^[[:space:]]*model-root[[:space:]]*=.*$|model-root = "/data/ComfyUI/models"|' \
        "${CONFIG_PATH}"
else
    printf '\nmodel-root = "/data/ComfyUI/models"\n' >>"${CONFIG_PATH}"
fi
if grep -Eq '^[[:space:]]*state-dir[[:space:]]*=' "${CONFIG_PATH}"; then
    sed -Ei \
        's|^[[:space:]]*state-dir[[:space:]]*=.*$|state-dir = "/data/krea2/state"|' \
        "${CONFIG_PATH}"
else
    printf '\nstate-dir = "/data/krea2/state"\n' >>"${CONFIG_PATH}"
fi
if grep -Eq '^[[:space:]]*service-mode[[:space:]]*=' "${CONFIG_PATH}"; then
    sed -Ei \
        's|^[[:space:]]*service-mode[[:space:]]*=.*$|service-mode = true|' \
        "${CONFIG_PATH}"
else
    printf '\nservice-mode = true\n' >>"${CONFIG_PATH}"
fi
if grep -Eq '^[[:space:]]*api-host[[:space:]]*=' "${CONFIG_PATH}"; then
    sed -Ei \
        's|^[[:space:]]*api-host[[:space:]]*=.*$|api-host = "127.0.0.1"|' \
        "${CONFIG_PATH}"
else
    printf '\napi-host = "127.0.0.1"\n' >>"${CONFIG_PATH}"
fi
if ! grep -Eq '^[[:space:]]*api-port[[:space:]]*=' "${CONFIG_PATH}"; then
    printf '\napi-port = 8787\n' >>"${CONFIG_PATH}"
fi
if ! grep -Eq \
    "^[[:space:]]*prompt-mode[[:space:]]*=[[:space:]]*['\"]theme['\"]" \
    "${CONFIG_PATH}" \
    && grep -Eq \
    '^[[:space:]]*(reconcile-interval|watch)[[:space:]]*=[[:space:]]*0([.]0*)?[[:space:]]*(#.*)?$' \
    "${CONFIG_PATH}"; then
    sed -Ei \
        's@^[[:space:]]*(reconcile-interval|watch)[[:space:]]*=.*$@reconcile-interval = 300@' \
        "${CONFIG_PATH}"
fi
chown "${SERVICE_USER}:${SERVICE_GROUP}" "${CONFIG_PATH}"
chmod 0660 "${CONFIG_PATH}"

runuser -u "${SERVICE_USER}" -- env \
    HOME="${APP_HOME}" \
    UV_CACHE_DIR="${APP_HOME}/.cache/uv" \
    "${APP_HOME}/bin/uv" sync \
        --project "${APP_HOME}" \
        --frozen \
        --no-dev

install -m 0644 "${SOURCE_ROOT}/deploy/krea2pipe.service" "${UNIT_PATH}"
install -m 0644 \
    "${SOURCE_ROOT}/deploy/krea2pipe-retry.service" "${RETRY_UNIT_PATH}"
systemctl daemon-reload

if [[ "${START_SERVICE}" -eq 1 ]]; then
    systemctl enable --now "${SERVICE_NAME}.service"
    if ! systemctl is-active --quiet "${SERVICE_NAME}.service"; then
        systemctl --no-pager --full status "${SERVICE_NAME}.service" || true
        fail "${SERVICE_NAME}.service did not become active."
    fi
    systemctl --no-pager --full status "${SERVICE_NAME}.service"
else
    printf 'Installed %s without starting it.\n' "${SERVICE_NAME}.service"
    printf 'Start it with: systemctl enable --now %s.service\n' "${SERVICE_NAME}"
fi

printf 'Configuration: %s\n' "${CONFIG_PATH}"
printf 'State directory: %s\n' "${APP_HOME}/state"
printf 'Prompt directory: %s\n' "${APP_HOME}/prompts"
printf 'Output directory: %s\n' "${APP_HOME}/output"
