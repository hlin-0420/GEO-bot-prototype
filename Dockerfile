# Use an official lightweight Python image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the application files
COPY . /app

# Ensure the templates folder is included
COPY templates /app/templates

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Set the command to run the Flask app
CMD ["python", "offline-app.py"]