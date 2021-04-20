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
	
