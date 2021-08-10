clean:
	pipenv --rm

init: 
	pipenv install Pipfile --dev

format:
	pipenv run python -m isort . --atomic
	pipenv run python -m black

jupyter-notebook:
	pipenv run jupyter notebook