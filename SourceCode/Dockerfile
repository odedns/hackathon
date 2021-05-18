
FROM python:3
WORKDIR /usr/src/app
COPY hackathon ./
RUN apt-get update -y && apt-get -y install graphviz libgraphviz-dev
RUN pip install numpy pandas matplotlib flask setuptools pygraphviz jupyter 
RUN pip install --upgrade azure-ai-anomalydetector
RUN cd SourceCode/src && ls && python3  ./setup.py build && python3 ./setup.py install 
WORKDIR /usr/src/app
CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root 

