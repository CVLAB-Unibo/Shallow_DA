#%%
import glob

fout = open("/data/aCardace/iterativive_da/splits/gta/train_dacs.txt", "w")
c=0
for i, fname in enumerate(glob.glob("/data/aCardace/datasets/GTA5/step1_dacs_da/*")):
    
    image_name = fname.replace("/data/aCardace/datasets/", "")
    number = fname.split("/")[-1].replace(".png", "")
    if len(number)>=5:
        c+=1
        label_name = image_name.replace("step1_dacs_da", "step1_dacs_da_semantic_encoded").replace(".png", "_encoded.png")
        line = f"{image_name};{label_name};{image_name}\n"
        fout.write(line)
    else:
        if int(number)<2975:
            c+=1
            label_name = image_name.replace("step1_dacs_da", "step1_dacs_da_semantic_encoded")
            line = f"{image_name};{label_name};{image_name}\n"
            fout.write(line)

print("found ", c)
fout.close()
#%%


