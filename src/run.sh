#python main.py #--work_mode translation


# p=0: no perturb / p=1: yes perturb
# s=0: scheme 0 (Tuan) / s=2: scheme 2 (2005 paper)

#python test_procrustes.py -p 0 -s 0
#python test_procrustes.py -p 0 -s 2
python test_procrustes.py -p 1 -s 0
#python test_procrustes.py -p 1 -s 2
