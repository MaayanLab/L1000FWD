FROM ipython/scipystack

MAINTAINER Zichen Wang <zichen.wang@mssm.edu>

# Copy the application folder inside the container
ADD requirements.txt /notebook/requirements.txt

# Set the default directory where CMD will execute
WORKDIR /notebook

RUN apt-get update -qq && \
	apt-get install -y python-mysqldb

RUN pip2 install -r requirements.txt

# Set environment variable
ENV HOME /notebook

EXPOSE 8888

ADD . /notebook

ADD notebook.sh /

# Start notebook server
CMD ["/notebook.sh"]
