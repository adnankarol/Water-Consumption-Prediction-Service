# Pull Python Image from Docker Hub
FROM python:3.7.7

MAINTAINER adnan karol <adnanmushtaq5@gmail.com>

# Create a working directory in a container and copy all files to the directory
WORKDIR /app
COPY . /app

# Install all the dependencies in the Docker Container
RUN pip3 install -r requirements.txt

ENV LISTEN_PORT=5000
EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["app.py"]
