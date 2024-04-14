from keras.datasets import mnist

(training_x, training_y), (test_x, test_y) = mnist.load_data()

print(training_x)

with open('./mnist_training.dat', 'w') as f:
    f.writelines([
        f"{len(training_x)} {len(training_y)}\n",
        f"{len(training_x[0])} {len(training_x[0][0])}\n"])
    
    for data in training_x:
        for r in data:
            for c in r:
                f.write(f"{c} ")

        f.write("\n")

    for data in training_y:
        f.write(f"{data} ")

    f.write("\n")

    
