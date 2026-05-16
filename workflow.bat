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
:: - hdfs already contains the 2GB raw data in /big-data-project-data/ecommerce.csv
:: - preprocessing.py saves the preprocessed data to jupyter:/preprocessed-data/part-00000-_____.csv
:: - machine-learning.py saves the result in <----------
:: - visualization.py saves the results to jupyter:/visualization-results/
:: - all the produced files are copied to the local directory: results

:: intialize containers
cd compose-images && docker compose up -d && cd ..

:: move data and scripts to containers
:: we assume the raw data to be in HDFS (to save transfer time)
docker cp scripts/preprocessing.py jupyter:/home/jovyan/
docker cp scripts/machine-learning.py jupyter:/home/jovyan/
docker cp scripts/visualization.py jupyter:/home/jovyan/

:: if create a brand-new results directory
rm -r results
mkdir results

:: preprocesing
docker exec -it jupyter python /home/jovyan/preprocessing.py
docker cp jupyter:/home/jovyan/preprocessed-data/part-00000-_______ results/preprocessed-data.csv

:: machine learning
docker exec -it jupyter python /home/jovyan/machine-learning.py
:: what are the steps? docker cp jupyter:/home/jovyan/

:: visualization
docker exec -it jupyter python /home/jovyan/visualization.py
docker cp jupyter:/home/jovyan/visualization-results/* results/

:: compose down containers
cd compose-images && docker compose down && cd ..
