# initialization
cd compose-images && docker compose up -d && cd ..

# copy notebooks to container
docker cp notebooks/preprocessing.ipynb jupyter:/home/jovyan/
docker cp notebooks/machine-learning.ipynb jupyter:/home/jovyan/
docker cp notebooks/visualization.ipynb jupyter:/home/jovyan/

# start jupyter notebook server
docker exec -it jupyter jupyter server list
echo copy the token from above output
start http://localhost:8888

echo now work on the preprocessing notebook first
echo then machine-learning notebook
echo then visualization notebook
echo after finishing all notebooks, run notebook-workflow-2.sh
