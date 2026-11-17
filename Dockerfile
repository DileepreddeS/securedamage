# Clean, pinned environment: Ubuntu 24.04 ships Python 3.12 and the glibc the OpenFHE wheel needs.
FROM --platform=linux/amd64 ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-pip libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]
CMD ["--quick"]
