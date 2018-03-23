#!/bin/bash/
step=0
actual_step=0
train_options=("pure_corpus" "mixed" "pure_oracle")
cnn_options=(1 2 3)
shuffle_options=(True False)
# in rainlf's pc, it should be "py -3.5 biLSTM_RNN.py"
python_exec_cmd="python biLSTM_RNN.py"


for train_option in ${train_options[*]}
do
	for cnn_option in ${cnn_options[*]}
	do
		for shuffle_option in ${shuffle_options[*]}
		do
			for iter in {1..10}
			do
				((step++))
				echo "step $step--------------"
				experi_data="tmp/experi_data/${train_option}_crf${cnn_option}iter${iter}${shuffle_option}biLSTMResult.csv"
				# The experiment data of this options is still not existed
				if [ ! -f "$experi_data" ]; then
					echo "excuting $step..."
					echo "$python_exec_cmd --train_option $train_option --crf_option $crf_option --iteration $iter --shuffle $shuffle_option"
					((actual_step++))
				else
					echo "experiment for [train_option=$train_option, cnn_option=$cnn_option, shuffle_option=$shuffle_option] \n already exists, so skip it..."
				fi
				echo ""
			done
		done
	done
done

echo "All finished, $actual_step of $ $step steps have been truely excecuted"