mkdir results/svc-supervise-change
mkdir results/svc-dagger-change
mkdir results/dt-dagger-change
mkdir results/dt-sup-change
mkdir data/svc-supervise-change
mkdir data/svc-dagger-change
mkdir data/dt-dagger-change
mkdir data/dt-sup-change

python python/competition/testbed-dt-sup-change.py
python python/competition/testbed-dt-dagger-change.py
python python/competition/testbed-svc-sup-change.py
python python/competition/testbed-svc-dagger-change.py
