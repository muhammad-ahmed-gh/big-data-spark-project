:: assumptions
:: - folder structure contains:
:: |__ scripts/
:: |   |__ preprocessing.py
:: |   |__ machine-learning.py
:: |   |__ visualization.py
:: |__ notebooks/
:: |   |__ preprocessing.ipynb
:: |   |__ machine-learning.ipynb
:: |   |__ visualization.ipynb
:: |__ results/
:: |__ compose-images/
:: |__ workflow.sh
::
:: - jupyter container folder structure
:: /home/jovyan/
:: |__ preprocessing.py
:: |__ machine-learning.py
:: |__ visualization.py
:: |__ preprocessing-result/
:: |__ machine-learning-result/
:: |__ visualization-result/
::
:: - hdfs already contains the 2GB raw data in /big-data-project-data/ecommerce.csv
:: - preprocessing.py saves the preprocessed data to jupyter:/home/jovyan/preprocessing-result/part-00000-_____.csv
:: - machine-learning.py saves the results to jupyter:/home/jovyan/machine-learning-result/*
:: - visualization.py saves the results to jupyter:/home/jovyan/visualization-result/*.png
:: - all the produced files are copied to the local directory: results

:: intialize containers
cd compose-images && docker compose up -d && cd ..

:: move data and scripts to containers
:: we assume the raw data to be in HDFS (to save transfer time)
docker cp scripts/preprocessing.py jupyter:/home/jovyan/
docker cp scripts/machine-learning.py jupyter:/home/jovyan/
docker cp scripts/visualization.py jupyter:/home/jovyan/

:: if create a brand-new results directory
rm results/*

:: preprocesing
docker exec -it jupyter spark-submit /home/jovyan/preprocessing.py
docker cp jupyter:/home/jovyan/preprocessing-result/. results

:: machine learning
docker exec -it jupyter spark-submit /home/jovyan/machine-learning.py
docker cp jupyter:/home/jovyan/machine-learning-result/. results

:: visualization
docker exec -it jupyter spark-submit /home/jovyan/visualization.py
docker cp jupyter:/home/jovyan/visualization-result/. results

:: compose down containers
cd compose-images && docker compose down && cd ..
