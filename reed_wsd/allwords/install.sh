sudo apt update && sudo apt upgrade
sudo apt install curl
[ ! -d "data" ] &&  mkdir -p "data"
cd data
curl -o raganato.zip http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
unzip raganato.zip
rm raganato.zip
cd ..
echo 'Compiling XML files into JSON.'
python3 raganato.py data/WSD_Evaluation_Framework/ data/raganato.json
echo 'Compiling contextualized word vectors.'
[ ! -d "data/vecs" ] &&  mkdir -p "data/vecs"
python3 vectorize.py data/raganato.json data/vecs/
echo 'Done!'
