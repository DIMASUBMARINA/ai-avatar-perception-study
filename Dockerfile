FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "python src/benchmark.py && python src/process_data.py && python src/participant_level.py && python src/visualize.py"]
