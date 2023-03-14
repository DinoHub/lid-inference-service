build:
	docker build . -t gradio-lid-inference-service:1.0.0
dev:
	docker run -it --entrypoint /bin/bash --gpus all gradio-lid-inference-service:1.0.0
