step=0

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_corpus --crf_option 1 --iteration $i --shuffle False
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_corpus --crf_option 2 --iteration $i --shuffle False
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_corpus --crf_option 3 --iteration $i --shuffle False
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option mixed --crf_option 1 --iteration $i --shuffle False
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option mixed --crf_option 2 --iteration $i --shuffle False
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option mixed --crf_option 3 --iteration $i --shuffle False
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_oracle --crf_option 1 --iteration $i --shuffle False
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_oracle --crf_option 2 --iteration $i --shuffle False
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_oracle --crf_option 3 --iteration $i --shuffle False
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_corpus --crf_option 1 --iteration $i --shuffle True
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_corpus --crf_option 2 --iteration $i --shuffle True
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_corpus --crf_option 3 --iteration $i --shuffle True
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option mixed --crf_option 1 --iteration $i --shuffle True
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option mixed --crf_option 2 --iteration $i --shuffle True
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option mixed --crf_option 3 --iteration $i --shuffle True
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_oracle --crf_option 1 --iteration $i --shuffle True
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_oracle --crf_option 2 --iteration $i --shuffle True
   echo "step ${step} finished".
done

for i in {1..10}
do
   ((step++))
   python biLSTM_RNN.py --train_option pure_oracle --crf_option 3 --iteration $i --shuffle True
   echo "step ${step} finished".
done

echo "all finished"