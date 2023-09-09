# Use the official lightweight Python image.
FROM python:3.11

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user and switch to it
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set work directory
WORKDIR $APP_HOME

# Change ownership of app files to the new user
COPY --chown=user . $HOME/app

# Run the web service on container startup.
CMD exec gunicorn --bind 0.0.0.0:7860 --workers 9 --threads 16 --timeout 120 main:app

