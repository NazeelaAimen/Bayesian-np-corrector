N = (126 2XX 512)

for n in ${N[*]}; do
	sbacth --parsable slurm_submit.sh n 
done


