build-image:
	 docker build -t babyandy0111/search-image-server-api:latest .

docker-run:
	docker run --name image-search -p 8000:8000 babyandy0111/search-image-server-api:latest

exec-run:
	docker exec -it image-search /bin/bash

run-build:
	make build-image && make docker-run

run-server:
	uvicorn api:app --host 0.0.0.0 --port 8000