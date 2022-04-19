# LINTING
lint:
	yapf -i src/*.py

# EVALUATE
ioueval:
	python3 src/iou.py \
	--iou 0.4 \
	--area 0.8

deteval:
	python3 src/deteval.py \
	--tp 0.4 \
	--tr 0.8

clear-asset:
	rm -rf out/*

# TRAINING
train: lint
	python3 src/train.py

train-synth: lint
	python3 src/train_synth.py
