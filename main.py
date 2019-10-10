import click
import yaml

from pipeline.pipeline_manager import pipeline_manager

@click.group()
def main():
	pass

@main.command()
@click.option("--model_name",'-n',required=False)
def train(model_name=None):
	if model_name is not None:
		print(model_name)
	manager.train(model_name)

@main.command()
@click.option("--model_name",'-n',required=False)
def test(model_name=None):
	if model_name is not None:
		print(model_name)
	manager.test(model_name)

@main.command()
@click.option("--model_name",'-n',required=False)
def test_dir(model_name=None):
	if model_name is not None:
		print(model_name)
	manager.test_dir(model_name)



@main.command()
@click.option("--model_path",'-m',required=True)
@click.option("--img_path",'-i',required=True)
@click.option("--res_path",'-r',required=True)
def detect(model_path,img_path,res_path):

	manager.detect(model_path,img_path,res_path)




if __name__ == '__main__':
	config=yaml.safe_load(open("config/config.yaml","r"))
	manager=pipeline_manager(config)
	main()
