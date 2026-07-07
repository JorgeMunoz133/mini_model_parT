import uproot

files = ["datasets/ttHTobb.root", "datasets/ttHTocc.root", "datasets/qcd_bctoe.root"]

for f in files:
    tree = uproot.open(f)["Events"]
    print(f, tree.num_entries)
