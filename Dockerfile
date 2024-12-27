FROM python:3.11-bullseye

RUN apt-get update && apt-get install -y \
git \
curl \
&& rm -rf /var/lib/apt/lists/*

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
&& unzip awscliv2.zip \
&& ./aws/install \
&& rm -rf aws awscliv2.zip

WORKDIR /app
COPY . .
RUN pip uninstall -y numpy pandas
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x infrastructure/deploy.sh infrastructure/update.sh

ENV PYTHONUNBUFFERED=1
ENV AWS_DEFAULT_REGION=us-west-2

CMD ["python", "src/run_evaluation.py"]