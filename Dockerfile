# Use an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
# Using --no-cache-dir is standard practice in Docker to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Define the command to run your app
# Assuming your main file is named 'app.py'. If it's 'stock_forecasting_app.py', rename it here.
CMD ["streamlit", "run", "app.py"]
