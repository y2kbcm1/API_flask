import argparse



parser = argparse.ArgumentParser(description='Train and save a model over the PIMA Indian Diabete')
parser.add_argument('integer', metavar='N', type=int, nargs=1,
                    help='an integer for the grid search')
parser.add_argument('filename', metavar='namefile', type=str, nargs=1,
                    help='filename')
args = parser.parse_args()

print(args.integer[0])
print(args.filename[0])