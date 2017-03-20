# # python /notebooks/Networks/CIFAR_10_Student_Networks/cifar_student.py -N 1 -H 32 -M baseline
# # python /notebooks/Networks/CIFAR_10_Student_Networks/cifar_student.py -N 2 -H 32 -M baseline
# # python /notebooks/Networks/CIFAR_10_Student_Networks/cifar_student.py -N 2 -H 64 -M baseline

# # python /notebooks/Networks/CIFAR_10_Student_Networks/cifar_student.py -N 1 -H 32 -M reg_logits
# # python /notebooks/Networks/CIFAR_10_Student_Networks/cifar_student.py -N 2 -H 32 -M reg_logits
# # python /notebooks/Networks/CIFAR_10_Student_Networks/cifar_student.py -N 2 -H 64 -M reg_logits

# # python /notebooks/Networks/CIFAR_10_Student_Networks/cifar_student.py -N 1 -H 32 -M know_dist
# # python /notebooks/Networks/CIFAR_10_Student_Networks/cifar_student.py -N 2 -H 32 -M know_dist
# # python /notebooks/Networks/CIFAR_10_Student_Networks/cifar_student.py -N 2 -H 64 -M know_dist

# # python /notebooks/Networks/Top_Down_Student_Networks/student_1.py -K 10 -H 15 -M baseline
# # python /notebooks/Networks/Top_Down_Student_Networks/student_1.py -K 10 -H 15 -M reg_logits
# # python /notebooks/Networks/Top_Down_Student_Networks/student_1.py -K 10 -H 15 -M know_dist

# # python /notebooks/Networks/Top_Down_Student_Networks/student_1.py -K 15 -H 20 -M baseline
# # python /notebooks/Networks/Top_Down_Student_Networks/student_1.py -K 15 -H 20 -M reg_logits
# # python /notebooks/Networks/Top_Down_Student_Networks/student_1.py -K 15 -H 20 -M know_dist

python /notebooks/Networks/Top_Down_Student_Networks/student_1.py -K 7 -H 10 -M baseline
python /notebooks/Networks/Top_Down_Student_Networks/student_1.py -K 7 -H 10 -M reg_logits
python /notebooks/Networks/Top_Down_Student_Networks/student_1.py -K 7 -H 10 -M know_dist

# python /notebooks/Networks/Top_Down_Student_Networks/student_2.py -C 38 -H False -M baseline
# python /notebooks/Networks/Top_Down_Student_Networks/student_2.py -C 38 -H True -M baseline
# python /notebooks/Networks/Top_Down_Student_Networks/student_2.py -C 25 -H False -M baseline
# python /notebooks/Networks/Top_Down_Student_Networks/student_2.py -C 25 -H True -M baseline

# python /notebooks/Networks/Top_Down_Student_Networks/student_2.py -C 38 -H False -M know_dist
# python /notebooks/Networks/Top_Down_Student_Networks/student_2.py -C 38 -H True -M know_dist
# python /notebooks/Networks/Top_Down_Student_Networks/student_2.py -C 25 -H False -M know_dist
# python /notebooks/Networks/Top_Down_Student_Networks/student_2.py -C 25 -H True -M know_dist