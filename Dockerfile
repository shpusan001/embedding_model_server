FROM python:3.11-slim

# 시스템 패키지 설치 + Poetry 설치
RUN apt-get update && apt-get install -y curl git && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# 프로젝트 디렉터리
WORKDIR /app

# Poetry 설정
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# pyproject.toml + lock 파일 복사 및 종속성 설치
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# 코드 및 모델 복사
COPY main.py .
COPY local_models ./local_models

# 포트 열기
EXPOSE 8000

# 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
