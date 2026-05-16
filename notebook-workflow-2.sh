# finalization (assumes containers are still running and notebooks are complete)

# copy results from jupyter container to local results directory
rm results/notebook-results/*
docker cp jupyter:/home/jovyan/preprocesing-result/. results/notebook-results/
docker cp jupyter:/home/jovyan/machine-learning-result/. results/notebook-results/
docker cp jupyter:/home/jovyan/visualization-result/. results/notebook-results/

# stop all containers
cd compose-images && docker compose down && cd ..
