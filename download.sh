#!/bin/bash

echo "Creating embeddings folder"
mkdir -m 777 "embeddings"
echo "Done!!!"

echo "Downloading pre-trained embeddings"
wget http://143.107.183.175:23580/twitter_doc2vec.model
mv twitter_doc2vec.model embeddings/
wget http://143.107.183.175:23580/news_doc2vec.model
mv news_doc2vec.model embeddings/
echo "Done!!!"