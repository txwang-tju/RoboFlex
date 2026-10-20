FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY roboflex ./roboflex
RUN pip install --no-cache-dir . && useradd --create-home robot
USER robot
EXPOSE 8766
# Set ROBOFLEX_ACCESS_TOKEN. Put TLS/auth/rate limiting at the private gateway.
# Model credentials are provided at runtime, never copied into the image.
CMD ["python", "-m", "roboflex", "cloud", "--host", "0.0.0.0", "--provider", "live"]
