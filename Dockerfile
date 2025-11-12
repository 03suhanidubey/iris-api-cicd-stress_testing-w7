# 1. Use official Python base image
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy files into the container
COPY . /app 

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose the port the app runs on
EXPOSE 8200

# 6. Define the command to run the application
CMD ["uvicorn", "demo_log:app", "--host", "0.0.0.0", "--port", "8200"]
# Note: The CMD command should be in JSON array format for proper execution
# If you want to use shell form, it should be: CMD uvicorn iris_fast:app --host
