tensorboard_path="${PWD}/trained/"
echo "$tensorboard_path"
venv/Scripts/tensorboard.exe --logdir $tensorboard_path
sleep 5