FROM python:3.7.5-stretch

RUN apt-get update && apt-get install -y \
python3-dev \
build-essential    
        
WORKDIR /app

ADD . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80
EXPOSE 8080

ENV NAME AIWorkflow

CMD ["python", "app.py"]
