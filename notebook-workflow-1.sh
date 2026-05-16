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

# now work on the preprocessing notebook first
# then machine-learning notebook
# then visualization notebook
# after finishing all notebooks, run notebook-workflow-2.bat
