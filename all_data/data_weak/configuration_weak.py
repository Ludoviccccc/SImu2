import json
if __name__ == "__main__":
    folder = "data_weak"
    folder1module = "data_1module"
    image_folder ="image_weak"
    num_addr = 20
    N = int(10000)
    N_init = 1000
    max_len = 50
    periode = 100
    num_bank = 4 # m'est pas variable, a changer dans dossier simulation
    mutation_rate = .1
    ks = [1,2,3,4]

    config = {"N_init":N_init,
              "N":N,
              "mutation_rate":mutation_rate,
              "max_len":max_len,
              "num_addr":20,
              "num_bank":num_bank,
              "image_folder":image_folder,
              "folder":folder,
              "ks":ks}
    with open(f"{folder}/config.json","w") as f:
        json.dump(config, f)
