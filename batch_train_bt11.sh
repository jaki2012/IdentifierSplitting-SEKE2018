# future work expand
step=0
actual_step=0
train_options=("pure_corpus")
cnn_options=(1)
shuffle_options=(False)
# in rainlf's pc, it should be "py -3.5 biLSTM_RNN.py"
python_exec_cmd="python biLSTM_RNN.py"


for iter in {1..10}
do
	for cnn_option in ${cnn_options[*]}
	do
		for shuffle_option in ${shuffle_options[*]}
		do
			for train_option in ${train_options[*]}
			do
				((step++))
				echo "step $step--------------"
				experi_data="experi_data/hs_bt11/${train_option}_cnn${cnn_option}iter${iter}${shuffle_option}biLSTMResult.csv"
				# The experiment data of this options is still not existed
				if [ ! -f "$experi_data" ]; then
					echo "excuting $step..."
					$python_exec_cmd --train_option $train_option --cnn_option $cnn_option --iteration $iter --shuffle $shuffle_option
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
