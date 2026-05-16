:: initialization
cd compose-images && docker compose up -d && cd ..
docker cp notebooks/preprocessing.ipynb jupyter:/home/jovyan/
docker cp notebooks/machine-learning.ipynb jupyter:/home/jovyan/
docker cp notebooks/visualization.ipynb jupyter:/home/jovyan/

:: start jupyter notebook
docker exec -it jupyter jupyter server list
echo copy the token
start http://localhost:8888

:: now work on the preprocessing notebook and
:: then run the next shell script
