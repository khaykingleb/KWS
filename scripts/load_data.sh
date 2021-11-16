mkdir data
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -O data/speech_commands_v0.01.tar.gz
mkdir data/speech_commands && tar -C data/speech_commands -xvzf data/speech_commands_v0.01.tar.gz 1> log
rm data/speech_commands_v0.01.tar.gz
