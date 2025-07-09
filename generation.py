import os
import time
import pickle
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from config.config import _C as cfg
from utils import get_model



def print_shape_stats(configs):
    conv_field = configs.conv_layers + configs.conv_size // 2 # 21
    sample_dim = 40
    sample_x_padded = sample_dim * configs.sample_outpaint_ratio + 2 * conv_field * configs.boundary_layers # 82
    sample_y_padded = sample_dim * configs.sample_outpaint_ratio + 2 * conv_field * configs.boundary_layers # 82
    sample_z_padded = sample_dim * configs.sample_outpaint_ratio + conv_field * configs.boundary_layers # 61
    print(f"padded shape zyx: [{sample_z_padded}, {sample_y_padded}, {sample_x_padded}]")
    sample_batch_z = sample_z_padded + 1 * conv_field + 2
    sample_batch_y = sample_y_padded + 2 * conv_field + 1
    sample_batch_x = sample_x_padded + 2 * conv_field
    print(f"sample shape zyx: [{sample_batch_z}, {sample_batch_y}, {sample_batch_x}]")
    # range start, end
    zstart, zend = conv_field + 2, sample_z_padded + conv_field + 2
    ystart, yend = conv_field + 1, sample_y_padded + conv_field + 1
    xstart, xend = conv_field, sample_x_padded + conv_field
    print(f"generation range zyx: [{zstart}:{zend}, {ystart}:{yend}, {xstart}:{xend}]")
    # move range
    m_zstart = (configs.boundary_layers + 1) * conv_field + 2
    m_zend = sample_batch_z
    m_ystart = (configs.boundary_layers + 1) * conv_field + 1
    m_yend = -((configs.boundary_layers + 1) * conv_field)
    m_xstart = (configs.boundary_layers + 1) * conv_field
    m_xend = -((configs.boundary_layers + 1) * conv_field)
    print(f"move range zyx: [{m_zstart}:{m_zend}, {m_ystart}:{m_yend}, {m_xstart}:{m_xend}]")
    """padded shape zyx: [61, 82, 82]
sample shape zyx: [84, 125, 124]
generation range zyx: [23:84, 22:104, 21:103]
move range zyx: [44:84, 43:-42, 42:-42] ---> [40, 40, 40]"""


def generate_samples_gated(configs, dataDims, model):
    if configs.sample_generation_mode == 'serial':
        if configs.CUDA:
            torch.cuda.synchronize()
        time_ge = time.time()

        sample_x_padded = dataDims['sample x dim'] + 2 * dataDims['conv field'] * configs.boundary_layers
        sample_y_padded = dataDims['sample y dim'] + 2 * dataDims['conv field'] * configs.boundary_layers  # don't need to pad the bottom
        sample_z_padded = dataDims['sample z dim'] + dataDims['conv field'] * configs.boundary_layers
        sample_conditions = dataDims['num conditioning variables']

        batches = int(np.ceil(configs.n_samples / configs.sample_batch_size))
        # n_samples = sample_batch_size * batches
        sample = torch.zeros(configs.n_samples, dataDims['channels'], dataDims['sample z dim'],
                             dataDims['sample y dim'],
                             dataDims['sample x dim'])  # sample placeholder
        print('Generating {} Samples'.format(configs.n_samples))

        for batch in range(batches):  # can't do these all at once so we do it in batches
            print('Batch {} of {} batches'.format(batch + 1, batches))
            sample_batch = torch.FloatTensor(configs.sample_batch_size, dataDims['channels'] + sample_conditions,
                                             sample_z_padded + 1 * dataDims['conv field'] + 2,
                                             sample_y_padded + 2 * dataDims['conv field'] + 1,
                                             sample_x_padded + 2 * dataDims['conv field']) # needs to be explicitly padded by the convolutional field
            sample_batch.fill_(0)  # initialize with minimum value

            #   if configs.do_conditioning: # assign conditions so the model knows what we want
            #      for i in range(len(configs.generation_conditions)):
            #         sample_batch[:,1+i,:,:] = (configs.generation_conditions[i] - dataDims['conditional mean']) / dataDims['conditional std']
            print([sample_batch.shape, sample.shape]) # [torch.Size([1, 1, 105, 167, 166]), torch.Size([1, 1, 40, 40, 40])]
            if configs.CUDA:
                sample_batch = sample_batch.cuda()

            # generator.train(False)
            model.eval()
            # k range(23, 82+21+2=105), j range(22, 124+21+1=146), i range(21, 124+21=145)
            with torch.no_grad():  # we will not be updating weights
                for k in tqdm.tqdm(
                        range(dataDims['conv field'] + 2,
                              sample_z_padded + dataDims['conv field'] + 2)):  # for each pixel
                    for j in range(dataDims['conv field'] + 1, sample_y_padded + dataDims['conv field'] + 1):
                        for i in range(dataDims['conv field'], sample_x_padded + dataDims['conv field']):
                            # out = generator(sample_batch.float())
                            out_string = f"z {k - dataDims['conv field'] - 2}:{k + 1}, " \
                                + f"y {j - dataDims['conv field'] - 1}:{j + dataDims['conv field'] * (1 - 0) + 1}, " \
                                + f"x {i - dataDims['conv field']}:{i + dataDims['conv field'] + 1}\t"

                            out = model(sample_batch[:, :, k - dataDims['conv field'] - 2:k + 1,
                                        j - dataDims['conv field'] - 1:j + dataDims['conv field'] * (1 - 0) + 1,
                                        i - dataDims['conv field']:i + dataDims['conv field'] + 1].float())
                            # out = torch.reshape(out, (
                            #     out.shape[0], dataDims['classes'] + 1, dataDims['channels'], out.shape[-3],
                            #     out.shape[-2],
                            #     out.shape[-1]))  # reshape to select channels
                            out = torch.reshape(out, (
                                out.shape[0], dataDims['classes'], dataDims['channels'], out.shape[-3],
                                out.shape[-2],
                                out.shape[-1]))  # reshape to select channels
                            # print(out.shape) [1, 3, 1, 24, 44, 43]
                            probs = F.softmax(out[:, :, 0, -1, -dataDims['conv field'] - 1, dataDims['conv field']],
                                              dim=1).data  # the remove the lowest element (boundary) [1, 3]
                            #  print(probs.shape)
                            #  print(sample_batch.shape)
                            sample_batch[:, 0, k, j, i] = (torch.multinomial(probs, 1).float() + 1).squeeze(1) / \
                                                          dataDims['classes']  # convert output back to training space
                            atom_id = int(sample_batch[:, 0, k, j, i].clone().cpu().item()*dataDims['classes'] - 1)
                            if atom_id == 1:
                                out_string = out_string + f"Atom H @ ({k} {j} {i}): {atom_id}"
                                print(out_string)
                            elif atom_id == 2:
                                out_string = out_string + f"Atom O @ ({k} {j} {i}): {atom_id}"
                                print(out_string)
                            del out, probs

            for k in range(dataDims['channels']): # copy the valide generation subpart to 'sample'
                sample[batch * configs.sample_batch_size:(batch + 1) * configs.sample_batch_size, k, :, :,
                :] = sample_batch[:, k, (configs.boundary_layers + 1) * dataDims['conv field'] + 2:,
                     (configs.boundary_layers + 1) * dataDims['conv field'] + 1:-(
                     (configs.boundary_layers + 1) * dataDims['conv field']),
                     (configs.boundary_layers + 1) * dataDims['conv field']:-(
                     (configs.boundary_layers + 1) * dataDims['conv field'])] * dataDims['classes'] - 1  # convert back to input space

        if configs.CUDA:
            torch.cuda.synchronize()
        time_ge = time.time() - time_ge


    return sample, time_ge


def generation(cfg, dataDims, model, epoch=-1):
    #err_te, time_te = test_net(model, te)  # clean run net

    sample, time_ge = generate_samples_gated(cfg, dataDims, model)  # generate samples
    np.save(os.path.join(cfg.OUTPUT_DIR, 'run_ep{}_samples_x{}'.format(epoch, cfg.sample_outpaint_ratio)), sample)

    if len(sample) != 0:
        print('Generated samples')

        #output_analysis = analyse_samples(sample)
        #agreements = compute_accuracy(cfg, dataDims, input_analysis, output_analysis)
        total_agreement = 0
       # for i, j, in enumerate(agreements.values()):
        #    if np.isnan(j) != 1: # kill NaNs
         #       total_agreement += float(j)

        #total_agreement /= len(agreements)

        #print('tot = {:.4f}; den={:.2f};time_ge={:.1f}s'.format(total_agreement, agreements['density'], time_ge))
        return sample, time_ge#, agreements, output_analysis

    else:
        print('Sample Generation Failed!')
        return 0, 0, 0, 0


def load_model_ckpt(ckpt_path: str):
    ckpt_ = torch.load(ckpt_path)
    cfg = ckpt_["cfg"]
    model = get_model(cfg)
    if 'module' in list(ckpt_["model_state_dict"].keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in ckpt_["model_state_dict"].items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(ckpt_["model_state_dict"])
    model= model.to("cuda" if cfg.CUDA else "cpu")
    model.eval()
    return model, ckpt_["epoch"]


def test_ckpt_generation(ckpt_path):
    dataDims_file = "datadims.pkl"
    if os.path.exists(dataDims_file):
        dataDims = pickle.load(open(dataDims_file, "rb"))
    else:
        raise FileNotFoundError
    # load model and ckpt
    model, epoch = load_model_ckpt(ckpt_path)
    generation(cfg, dataDims, model, epoch)
    print("done")


if __name__ == "__main__":
    test_ckpt_generation(ckpt_path="checkpoints/water-adam-lr1e-3/model-last-ep6.pt")
    # print_shape_stats(cfg)