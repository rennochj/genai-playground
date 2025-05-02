FROM cgr.dev/chainguard/python:latest-dev

USER root

RUN <<EOT

    apk update && apk add posix-libc-utils && ldconfig
    apk add zsh zsh-vcs curl wget git vim make docker docker-compose jq
    apk add sudo-rs shadow && echo "nonroot ALL = (ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers && echo y | pwck -q || true
    
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o ~/awscliv2.zip
    unzip ~/awscliv2.zip
    sudo ./aws/install --bin-dir /usr/bin --install-dir /usr/aws-cli --update

EOT

USER nonroot

RUN <<EOT

    pip install --upgrade pip

    # Install UV
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install default shell
    command -v zsh | sudo tee -a /etc/shells
    chsh nonroot --shell /usr/bin/zsh
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

EOT

COPY .devcontainer/dotfiles/* /home/nonroot

ENTRYPOINT ["/bin/sh","-c","sleep infinity"]
