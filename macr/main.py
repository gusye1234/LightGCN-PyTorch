import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
from utils import EarlyStopping
import wandb

wandb.init(project="LightGCN_leverage", entity="dain5832")
wandb.run.name = '{}_{}_{}_{}MACR'.format(world.dataset, world.config['weight_type'], world.config['norm_type'], world.config['exp'])
wandb.config.update(world.config)


Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
loss_name = getattr(utils, world.config['loss_type'])
loss_class = loss_name(Recmodel, world.config)
early_stopping = EarlyStopping(patience=world.config['patience'], verbose=True)

Neg_k = 1
if world.config['loss_type'] == 'BCELoss':
    Neg_k = 4

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], use_orig=False)
            if world.config['patience'] and epoch > 100:
                early_stopping(results['recall'][0], Recmodel)
                    
                if early_stopping.early_stop:
                    print("Early stopping")
                    break        

        output_information = Procedure.train_original(dataset, Recmodel, loss_class, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
    cprint("[FINAL TEST]")
    results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], use_orig=False, is_test=True)
    cprint("[TEST ORIG]")
    results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'], use_orig=True, is_test=True)

    # saved trained embeddings
    all_users, all_items = Recmodel.computer()
    torch.save(all_users, 'saved_emb/{}_{}_u_emb.pt'.format(world.dataset, world.config['latent_dim_rec']))
    torch.save(all_items, 'saved_emb/{}_{}_i_emb.pt'.format(world.dataset, world.config['latent_dim_rec']))

finally:
    if world.tensorboard:
        w.close()