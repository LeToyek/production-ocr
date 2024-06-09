FROM python:3.9-slim
WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && \
EXPOSE 5000
ENV FLASK_APP=app/main.py
CMD ["flask", "run", "--host=0.0.0.0"]
