:: it's assumed that the containers are running
:: and you've run all the notebooks

:: get data and compose down
docker cp jupyter:/home/jovyan/results/* results/
cd compose-images && docker compose down && cd ..
