# hackathon

Running the rest api:

cd SourceCode/src/hackathon

export FLASK_APP=rest

flask run

access from your browser :
http://localhost:5000/s2g?filename=<filename in DATA folder>&index=<column index in csv>&plen=<pattenr_len>&qlen=<query_len>

  
Running jupyter notebook as docker container:
1. create empty dir and cd to that dir
 
2. clone the project :
	git clone git@github.com:odedns/hackathon.git
3. build the docker image:
	docker build -f hackathon/SourceCode/Dockerfile -t odedns/hack .
4. run the docker image as a container:
	docker run -it --name hack --rm -p 8889:8888   odedns/hack 

5. access the jupyter notebook on your localhost at:
	http://localhost:8889
	enter the token given on the cmd line to login
	

if you want to mount your local dir into the docker container use:
docker run  -v <your local dir>:/usr/src/app/<name of dir> -it --name hack --rm -p 8889:8888   odedns/hack 
for example:
docker run  -v /home/oded/dev/python/docker/hackathon/SourceCode/example/:/usr/src/app/ex -it --name hack --rm -p 8889:8888   odedns/hack 

This will expose the example directory to jupyter running on docker as the ex directory.


Running the web application
---------------------------
1. install mongodb
2. load the appropriate data into mongo
3. create db:
 use hack
4. import the materna data from the DATA folder into mongo:
mongoimport -d hack -c materna1 --type csv --file materna-2-01.csv --headerline

5. build the web app container:
docker build -f hackathon/SourceCode/Dockerfile.flask -t odedns/hack_flask .

6. run the docker container:
docker run -it --name hack_flask --rm -p 5001:5000 -e MONGO_HOST=172.17.0.1 -e MONGO_PORT=27017 odedns/hack_flask 

7. access the webapp at:

http://localhost:5001/graph

