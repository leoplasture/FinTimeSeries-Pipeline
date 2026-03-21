FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
		PYTHONUNBUFFERED=1 \
		PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libraries used by scientific stack and plotting backends.
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
		gcc \
		g++ \
		libgomp1 \
		libfreetype6-dev \
		libpng-dev \
		curl \
		&& rm -rf /var/lib/apt/lists/*

# Cache-friendly dependency layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application source after dependencies are installed.
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
	CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/viz/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
