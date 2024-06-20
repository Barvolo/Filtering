FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make entrypoint.sh executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint to entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

