sudo apt update && sudo apt upgrade
sudo apt install curl
[ ! -d "data" ] &&  mkdir -p "data"
cd data
curl -o aclImdb_v1.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz
rm aclImdb_v1.tar.gz
cd ..
echo "preprocessing imdb data"
python3 preprocess.py
echo "Done!"

