

#python main.py -e cliptext -i cifar100 -w cifar100 -m nn
#python main.py -e cliptext -i cifar100 -w cifar100 -m csls_knn_10
python main.py -e cliptext -i cifar100 -w cifar100_category -m nn
python main.py -e fp -i cifar100 -w cifar100_category -m nn
#python main.py -e fp -i cifar100 -w cifar100_category -m nn
#python main.py -e fp -i cifar100 -w cifar100_category -m matching
#python main.py -e fp -i cifar100 -w cifar100 -m matching
#python main.py -e fp -i cifar10 -w cifar10 -m matching
