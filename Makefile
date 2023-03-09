build:
	docker build . -t gradio-lid-inference-service:1.0.0
dev:
	docker run -p 8085:8085 --rm -it -v ${PWD}:/dory gradio-lid-inference-service:1.0.0
