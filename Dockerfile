FROM python:2.7

# Get pip and install numpy/scipy dependencies
RUN apt-get update && apt-get install -y build-essential gfortran libatlas-base-dev default-libmysqlclient-dev

# Update pip
RUN pip install --upgrade pip

# Install required python packages
ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt


# Copy the application folder inside the container
ADD . /my_application
RUN chmod +x /my_application/boot.sh

# # Install required python packages
# RUN pip install -r /my_application/requirements.txt

# Expose ports
EXPOSE 5000

# Set the default directory where CMD will execute
WORKDIR /my_application

# Set the default command to execute
# when creating a new container
CMD /my_application/boot.sh
