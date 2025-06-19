FROM python:3.9-slim-buster 

# Set working directory 
WORKDIR /app 

# Copy your application code 

COPY . . 
# Copy requirements.txt and install dependencies 

COPY requirements.txt ./ 
RUN pip3 install -r requirements.txt 
RUN pip3 install gunicorn 

RUN mkdir -p /app/logs && chmod 755 /app/logs
# Expose port that Flask app uses 

EXPOSE 7860 

# Set the entrypoint to run your Flask app 

ENTRYPOINT ["python", "app.py"]
