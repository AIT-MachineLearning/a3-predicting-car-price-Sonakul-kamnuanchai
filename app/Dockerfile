# Use the official Python image from Docker Hub
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy everything from the project root into the container's '/app' directory
COPY . /app

# Install dependencies from 'requirements.txt'
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN pip install dash[testing]
RUN pip install pytest
RUN pip install pytest-depends
RUN pip install python-dotenv
# Command to run the app
CMD ["python", "app.py"]
