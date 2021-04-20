
FROM python:3
WORKDIR /usr/src/app
COPY hackathon ./
RUN pip install numpy pandas matplotlib flask setuptools jupyter 
RUN cd SourceCode/src && ls && python3  ./setup.py build && python3 ./setup.py install 
WORKDIR /usr/src/app
CMD /bin/sh

