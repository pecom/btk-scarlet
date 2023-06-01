import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gen_blend import main_single
import pandas as pd

if __name__ == "__main__":
    allshifts = np.array([76, 61, 54, 24, 0, 6])
    dframe_keys = ["cata_ndx", "noisy_image", "noiseless_image", "true_x", "true_y"]
    dframe_solo = {sk:[] for sk in dframe_keys}
     
    for i, sft in enumerate(allshifts):
        print(f"Processing index {i} object number {sft}")
        main_single(dframe_solo, sft)

    fullframe_solo = pd.DataFrame(data=dframe_solo)
    fullframe_solo.to_pickle(f'../output/single_ibright_blends.pkl')
    print("Αll done!")
