find directory -name './results/*.eps'|while read file; do
    mv $file ./results/svc-supervise-change/
done
