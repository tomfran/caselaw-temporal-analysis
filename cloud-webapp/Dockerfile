# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9.1

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR /illinois-cases-analysis-webapp
COPY . ./

# Install production dependencies.
RUN apt-get update &&\
    apt-get install -y binutils libproj-dev gdal-bin
RUN pip install -r requirements.txt

EXPOSE 8080

CMD cd webapp && python webapp.py