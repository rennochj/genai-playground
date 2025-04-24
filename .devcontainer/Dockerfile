FROM mcr.microsoft.com/devcontainers/python:3.12-bullseye

RUN <<EOT

    # Install dependencies
    apt update && apt upgrade -y
    apt install make curl

    # Install pip
    pip install --upgrade pip
    pip install poetry==1.7.1 mkdocs==1.5.3

    # Install AWS CLI v2
    curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install --bin-dir /usr/bin --install-dir /usr/aws-cli --update

EOT

USER vscode

RUN <<EOT

   # Install UV
   curl -LsSf https://astral.sh/uv/install.sh | sh

    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc

EOT