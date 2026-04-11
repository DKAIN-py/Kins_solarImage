FROM archlinux:latest

WORKDIR /fastapi

RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm python && \
    pacman -Scc --noconfirm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "server.py", "--host", "0.0.0.0", "--port", "8000"]